#!/usr/bin/env python3
"""
=============================================================================
  Timing Tracker  -  Layer / Row / Pipeline ETA System
=============================================================================

Provides granular timing instrumentation for AirLLM layer-by-layer inference:

  1. LayerTimer   - Times each transformer layer forward pass, predicts
                    remaining layers and total layer-pass completion.
  2. RowTimer     - Times each row's full generation cycle, predicts
                    remaining rows and total row completion.
  3. PipelineTimer - Wraps the entire pipeline with wall-clock tracking,
                     aggregates all metrics, exports timing_report.json.

Usage:
    from timing_tracker import LayerTimer, RowTimer, PipelineTimer

    pipeline_timer = PipelineTimer(total_rows=100)
    row_timer = pipeline_timer.row_timer
    layer_timer = pipeline_timer.layer_timer

    pipeline_timer.start()
    for row_idx in range(100):
        row_timer.start_row(row_idx)
        layer_timer.reset_for_new_generation()
        # ... AirLLM generates, calling layer_timer.record_layer() per layer ...
        row_timer.end_row(row_idx)
    pipeline_timer.finish()

=============================================================================
"""

import json
import time
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

log = logging.getLogger("airllm-pipeline")


# ---------------------------------------------------------------------------
# Utility: Human-readable duration
# ---------------------------------------------------------------------------
def format_duration(seconds: float) -> str:
    """Convert seconds to a human-readable string."""
    if seconds < 0:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.0f}s"


def format_timestamp(epoch: float) -> str:
    """Convert epoch to HH:MM:SS."""
    return time.strftime("%H:%M:%S", time.localtime(epoch))


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  LAYER TIMER                                                           ║
# ║  Times each transformer layer's forward pass during generation.        ║
# ╚═════════════════════════════════════════════════════════════════════════╝
@dataclass
class LayerTimer:
    """Tracks per-layer timing within a single generation call.

    AirLLM processes one transformer layer at a time. This timer records
    how long each layer takes, computes a running average, and predicts
    when all layers in the current forward pass will complete.

    For a 72B model like Qwen2.5-72B-Instruct, there are ~80 layers.
    Each generation token requires a full pass through all layers.
    """

    total_layers: int = 80  # Updated dynamically if detected

    # Internal state
    _layer_times: List[float] = field(default_factory=list)
    _current_layer_start: float = 0.0
    _pass_start: float = 0.0
    _current_layer_idx: int = 0
    _total_passes: int = 0  # How many full forward passes completed
    _all_layer_times: List[float] = field(default_factory=list)  # Across all passes

    def reset_for_new_generation(self):
        """Call before each new row's generation starts."""
        self._layer_times = []
        self._current_layer_idx = 0
        self._pass_start = time.time()

    def start_layer(self, layer_idx: int):
        """Call when a layer begins processing."""
        self._current_layer_idx = layer_idx
        self._current_layer_start = time.time()

    def end_layer(self, layer_idx: int):
        """Call when a layer finishes processing."""
        elapsed = time.time() - self._current_layer_start
        self._layer_times.append(elapsed)
        self._all_layer_times.append(elapsed)

        # Log every 10th layer or the last layer to avoid spam
        layers_done = len(self._layer_times)
        if layers_done % 10 == 0 or layers_done == self.total_layers:
            avg = statistics.mean(self._layer_times)
            remaining = self.total_layers - layers_done
            eta_secs = remaining * avg
            log.info(
                "    Layer %d/%d | this=%.2fs | avg=%.2fs | "
                "remaining=%d | ETA this pass: %s",
                layers_done, self.total_layers,
                elapsed, avg, remaining, format_duration(eta_secs),
            )

    def record_layer(self, layer_idx: int, elapsed: float):
        """Alternative: record a layer timing directly (no start/end calls)."""
        self._layer_times.append(elapsed)
        self._all_layer_times.append(elapsed)
        self._current_layer_idx = layer_idx

        layers_done = len(self._layer_times)
        if layers_done % 10 == 0 or layers_done == self.total_layers:
            avg = statistics.mean(self._layer_times)
            remaining = self.total_layers - layers_done
            eta_secs = remaining * avg
            log.info(
                "    Layer %d/%d | this=%.2fs | avg=%.2fs | "
                "remaining=%d | ETA this pass: %s",
                layers_done, self.total_layers,
                elapsed, avg, remaining, format_duration(eta_secs),
            )

    def complete_pass(self):
        """Mark a full forward pass as complete."""
        self._total_passes += 1
        if self._layer_times:
            pass_time = sum(self._layer_times)
            log.info(
                "    Pass #%d complete: %d layers in %s (avg %.2fs/layer)",
                self._total_passes, len(self._layer_times),
                format_duration(pass_time),
                statistics.mean(self._layer_times),
            )

    @property
    def avg_layer_time(self) -> float:
        """Average time per layer across ALL passes."""
        if not self._all_layer_times:
            return 0.0
        return statistics.mean(self._all_layer_times)

    @property
    def estimated_pass_time(self) -> float:
        """Estimated time for one full forward pass through all layers."""
        return self.avg_layer_time * self.total_layers

    def get_stats(self) -> Dict[str, Any]:
        """Return layer timing statistics."""
        if not self._all_layer_times:
            return {"total_layers_processed": 0}
        return {
            "total_layers_processed": len(self._all_layer_times),
            "total_passes": self._total_passes,
            "avg_layer_time_sec": round(self.avg_layer_time, 4),
            "min_layer_time_sec": round(min(self._all_layer_times), 4),
            "max_layer_time_sec": round(max(self._all_layer_times), 4),
            "median_layer_time_sec": round(statistics.median(self._all_layer_times), 4),
            "estimated_pass_time": format_duration(self.estimated_pass_time),
            "total_layers_per_model": self.total_layers,
        }


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  ROW TIMER                                                             ║
# ║  Times each row's full generation cycle and predicts remaining rows.   ║
# ╚═════════════════════════════════════════════════════════════════════════╝
@dataclass
class RowTimer:
    """Tracks per-row generation timing and predicts total completion.

    Records how long each row takes to generate, maintains a rolling
    average, and provides ETAs for remaining rows and total pipeline.
    """

    total_rows: int = 100

    # Internal state
    _row_times: Dict[int, float] = field(default_factory=dict)
    _row_start_times: Dict[int, float] = field(default_factory=dict)
    _rows_completed: int = 0
    _generation_start: float = 0.0

    def start_generation_phase(self):
        """Call once when the generation loop begins."""
        self._generation_start = time.time()

    def start_row(self, row_idx: int):
        """Call when a row begins processing."""
        self._row_start_times[row_idx] = time.time()

    def end_row(self, row_idx: int, output_tokens: int = 0):
        """Call when a row finishes processing."""
        if row_idx not in self._row_start_times:
            return

        elapsed = time.time() - self._row_start_times[row_idx]
        self._row_times[row_idx] = elapsed
        self._rows_completed += 1

        # Compute stats
        completed_times = list(self._row_times.values())
        avg_time = statistics.mean(completed_times)
        remaining_rows = self.total_rows - self._rows_completed
        eta_remaining = remaining_rows * avg_time
        eta_finish = time.time() + eta_remaining

        # Speed metrics
        total_elapsed = time.time() - self._generation_start
        rows_per_hour = (self._rows_completed / total_elapsed) * 3600 if total_elapsed > 0 else 0
        tokens_per_sec = output_tokens / elapsed if elapsed > 0 and output_tokens > 0 else 0

        # Progress bar
        pct = (self._rows_completed / self.total_rows) * 100
        bar_width = 30
        filled = int(bar_width * self._rows_completed / self.total_rows)
        bar = "█" * filled + "░" * (bar_width - filled)

        log.info(
            "  │ Row %d/%d done in %s",
            self._rows_completed, self.total_rows, format_duration(elapsed),
        )
        log.info(
            "  │ [%s] %.1f%%",
            bar, pct,
        )
        log.info(
            "  │ Avg/row: %s | Remaining: %d rows | ETA: %s (%s)",
            format_duration(avg_time), remaining_rows,
            format_duration(eta_remaining), format_timestamp(eta_finish),
        )
        log.info(
            "  │ Speed: %.1f rows/hr%s | Elapsed: %s",
            rows_per_hour,
            f" | {tokens_per_sec:.1f} tok/s" if tokens_per_sec > 0 else "",
            format_duration(total_elapsed),
        )
        log.info("  └" + "─" * 58)

    def skip_row(self, row_idx: int):
        """Mark a row as skipped (from checkpoint)."""
        self._rows_completed += 1

    @property
    def avg_row_time(self) -> float:
        """Average time per row."""
        if not self._row_times:
            return 0.0
        return statistics.mean(self._row_times.values())

    @property
    def total_elapsed(self) -> float:
        """Total time since generation phase started."""
        if self._generation_start == 0:
            return 0.0
        return time.time() - self._generation_start

    def get_stats(self) -> Dict[str, Any]:
        """Return row timing statistics."""
        if not self._row_times:
            return {"rows_completed": 0}

        times = list(self._row_times.values())
        return {
            "rows_completed": self._rows_completed,
            "total_rows": self.total_rows,
            "avg_row_time": format_duration(statistics.mean(times)),
            "avg_row_time_sec": round(statistics.mean(times), 2),
            "min_row_time": format_duration(min(times)),
            "min_row_time_sec": round(min(times), 2),
            "max_row_time": format_duration(max(times)),
            "max_row_time_sec": round(max(times), 2),
            "median_row_time": format_duration(statistics.median(times)),
            "median_row_time_sec": round(statistics.median(times), 2),
            "stdev_row_time_sec": round(statistics.stdev(times), 2) if len(times) > 1 else 0.0,
            "total_generation_time": format_duration(sum(times)),
            "total_generation_time_sec": round(sum(times), 2),
            "rows_per_hour": round((len(times) / sum(times)) * 3600, 1) if sum(times) > 0 else 0,
            "per_row_breakdown": {
                str(k): {"time": format_duration(v), "seconds": round(v, 2)}
                for k, v in sorted(self._row_times.items())
            },
        }


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE TIMER                                                        ║
# ║  Top-level orchestrator: wraps LayerTimer + RowTimer + wall clock.     ║
# ╚═════════════════════════════════════════════════════════════════════════╝
@dataclass
class PipelineTimer:
    """Top-level timer that orchestrates layer and row timing.

    Provides:
    - Wall-clock timing for the entire pipeline
    - Aggregated layer-level stats
    - Aggregated row-level stats with ETAs
    - JSON report export
    """

    total_rows: int = 100
    total_layers: int = 80
    report_path: str = "timing_report.json"

    # Sub-timers
    layer_timer: LayerTimer = field(default=None)
    row_timer: RowTimer = field(default=None)

    # Pipeline-level
    _pipeline_start: float = 0.0
    _pipeline_end: float = 0.0
    _phase_times: Dict[str, float] = field(default_factory=dict)
    _phase_starts: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.layer_timer is None:
            self.layer_timer = LayerTimer(total_layers=self.total_layers)
        if self.row_timer is None:
            self.row_timer = RowTimer(total_rows=self.total_rows)

    def start(self):
        """Start the pipeline timer."""
        self._pipeline_start = time.time()
        log.info(
            "\n┌%s┐", "─" * 58,
        )
        log.info(
            "│  TIMING TRACKER ACTIVE%s│",
            " " * 36,
        )
        log.info(
            "│  Rows: %-5d | Layers/model: %-5d%s│",
            self.total_rows, self.total_layers, " " * 16,
        )
        log.info(
            "│  Started: %s%s│",
            format_timestamp(self._pipeline_start), " " * 33,
        )
        log.info(
            "└%s┘", "─" * 58,
        )

    def start_phase(self, phase_name: str):
        """Start timing a named pipeline phase."""
        self._phase_starts[phase_name] = time.time()
        log.info("  ⏱  Phase '%s' started", phase_name)

    def end_phase(self, phase_name: str):
        """End timing a named pipeline phase."""
        if phase_name in self._phase_starts:
            elapsed = time.time() - self._phase_starts[phase_name]
            self._phase_times[phase_name] = elapsed
            log.info(
                "  ⏱  Phase '%s' completed in %s",
                phase_name, format_duration(elapsed),
            )

    def finish(self):
        """Finalize the pipeline timer and print summary."""
        self._pipeline_end = time.time()
        total = self._pipeline_end - self._pipeline_start

        log.info("\n" + "=" * 60)
        log.info("  TIMING REPORT")
        log.info("=" * 60)

        # Phase breakdown
        log.info("\n  Phase Breakdown:")
        for phase, elapsed in self._phase_times.items():
            pct = (elapsed / total) * 100 if total > 0 else 0
            log.info(
                "    %-25s %10s  (%5.1f%%)",
                phase, format_duration(elapsed), pct,
            )

        # Layer stats
        layer_stats = self.layer_timer.get_stats()
        if layer_stats.get("total_layers_processed", 0) > 0:
            log.info("\n  Layer Stats:")
            log.info("    Total layers processed:  %d", layer_stats["total_layers_processed"])
            log.info("    Total forward passes:    %d", layer_stats["total_passes"])
            log.info("    Avg time/layer:          %.4fs", layer_stats["avg_layer_time_sec"])
            log.info("    Min time/layer:          %.4fs", layer_stats["min_layer_time_sec"])
            log.info("    Max time/layer:          %.4fs", layer_stats["max_layer_time_sec"])
            log.info("    Est. time/full pass:     %s", layer_stats["estimated_pass_time"])

        # Row stats
        row_stats = self.row_timer.get_stats()
        if row_stats.get("rows_completed", 0) > 0:
            log.info("\n  Row Stats:")
            log.info("    Rows completed:          %d / %d", row_stats["rows_completed"], row_stats["total_rows"])
            log.info("    Avg time/row:            %s", row_stats["avg_row_time"])
            log.info("    Min time/row:            %s", row_stats["min_row_time"])
            log.info("    Max time/row:            %s", row_stats["max_row_time"])
            log.info("    Median time/row:         %s", row_stats["median_row_time"])
            log.info("    Throughput:              %.1f rows/hour", row_stats["rows_per_hour"])
            log.info("    Total generation time:   %s", row_stats["total_generation_time"])

        # Total
        log.info("\n  Total Pipeline Time:       %s", format_duration(total))
        log.info("  Started:                   %s", format_timestamp(self._pipeline_start))
        log.info("  Finished:                  %s", format_timestamp(self._pipeline_end))
        log.info("=" * 60)

        # Export JSON report
        self._export_report(total, layer_stats, row_stats)

    def _export_report(self, total_time: float, layer_stats: Dict, row_stats: Dict):
        """Export full timing report to JSON."""
        report = {
            "pipeline": {
                "total_time": format_duration(total_time),
                "total_time_sec": round(total_time, 2),
                "started": format_timestamp(self._pipeline_start),
                "finished": format_timestamp(self._pipeline_end),
            },
            "phases": {
                name: {
                    "time": format_duration(elapsed),
                    "seconds": round(elapsed, 2),
                    "percent": round((elapsed / total_time) * 100, 1) if total_time > 0 else 0,
                }
                for name, elapsed in self._phase_times.items()
            },
            "layers": layer_stats,
            "rows": row_stats,
        }

        report_path = Path(self.report_path)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        log.info("  Timing report saved: %s", report_path)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  AirLLM LAYER HOOK                                                     ║
# ║  Monkey-patches AirLLM's generate() to capture per-layer timing.      ║
# ╚═════════════════════════════════════════════════════════════════════════╝
def install_layer_hooks(model, layer_timer: LayerTimer) -> bool:
    """Attempt to install timing hooks on AirLLM's internal layer processing.

    AirLLM loads layers one at a time. This function tries to wrap the
    internal _run_layer or similar method to capture per-layer timing.

    Returns True if hooks were installed, False otherwise.
    """
    hooked = False

    # Strategy 1: Hook into the model's layer iteration
    # AirLLM typically has a method that iterates through layers
    for attr_name in ["_run_layer", "run_layer", "_forward_layer"]:
        if hasattr(model, attr_name):
            original_fn = getattr(model, attr_name)

            def make_wrapper(orig, timer):
                layer_counter = [0]

                def wrapper(*args, **kwargs):
                    start = time.time()
                    result = orig(*args, **kwargs)
                    elapsed = time.time() - start
                    timer.record_layer(layer_counter[0], elapsed)
                    layer_counter[0] += 1
                    if layer_counter[0] >= timer.total_layers:
                        timer.complete_pass()
                        layer_counter[0] = 0
                    return result
                return wrapper

            setattr(model, attr_name, make_wrapper(original_fn, layer_timer))
            log.info("  Layer hook installed on: %s.%s", type(model).__name__, attr_name)
            hooked = True
            break

    # Strategy 2: Hook into PyTorch module forward calls
    if not hooked:
        try:
            import torch.nn as nn
            # Try to find the transformer layers list
            for attr_name in ["layers", "model_layers", "h", "blocks"]:
                layers_list = getattr(model, attr_name, None)
                if layers_list is not None and isinstance(layers_list, (list, nn.ModuleList)):
                    layer_timer.total_layers = len(layers_list)
                    log.info(
                        "  Detected %d transformer layers via '%s'",
                        len(layers_list), attr_name,
                    )
                    # We can't easily hook ModuleList in AirLLM since layers
                    # are loaded/unloaded dynamically, but we detected the count
                    break
        except ImportError:
            pass

    if not hooked:
        log.info(
            "  Layer hooks not installed (AirLLM manages layers internally).\n"
            "  Row-level and pipeline-level timing will still work."
        )

    return hooked


def detect_layer_count(model_id: str) -> int:
    """Estimate the number of transformer layers based on model ID."""
    model_lower = model_id.lower()

    # Known layer counts for popular models
    layer_map = {
        "qwen2.5-72b": 80,
        "qwen2.5-32b": 64,
        "qwen2.5-14b": 48,
        "qwen2.5-7b": 32,
        "qwen2.5-3b": 36,
        "qwen2.5-1.5b": 28,
        "qwen2.5-0.5b": 24,
        "llama-3.1-70b": 80,
        "llama-3.1-8b": 32,
        "llama-3-70b": 80,
        "llama-3-8b": 32,
        "mixtral-8x22b": 56,
        "mixtral-8x7b": 32,
        "mistral-7b": 32,
        "falcon-180b": 80,
        "falcon-40b": 60,
    }

    for pattern, count in layer_map.items():
        if pattern in model_lower:
            log.info("  Detected %d layers for model: %s", count, model_id)
            return count

    log.info("  Unknown model architecture, defaulting to 80 layers")
    return 80
