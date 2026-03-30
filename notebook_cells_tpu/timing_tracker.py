from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


def format_duration(seconds: float) -> str:
    seconds = float(seconds or 0.0)
    if seconds < 60:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {s:.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {s:.0f}s"


def detect_layer_count(model_id: str) -> int:
    """
    Best-effort layer count from HF config.
    Returns 0 if unknown (never hard-fails).
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        for k in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            v = getattr(cfg, k, None)
            if isinstance(v, int) and v > 0:
                return v
    except Exception:
        pass
    return 0


@dataclass
class RowTimer:
    parent: "PipelineTimer"
    _row_start: Optional[float] = None
    _generation_phase_start: Optional[float] = None

    def start_generation_phase(self):
        self._generation_phase_start = time.perf_counter()

    def start_row(self, idx: int):
        self._row_start = time.perf_counter()

    def end_row(self, idx: int, output_tokens: int = 0):
        if self._row_start is None:
            return
        dt = time.perf_counter() - self._row_start
        self.parent.rows_done += 1
        self.parent.total_row_time_s += dt
        self.parent.total_output_tokens += max(0, int(output_tokens or 0))
        self.parent.row_metrics.append(
            {"row_index": int(idx), "duration_s": dt, "output_tokens": int(output_tokens or 0), "status": "done"}
        )
        self._row_start = None

    def skip_row(self, idx: int):
        self.parent.row_metrics.append(
            {"row_index": int(idx), "duration_s": 0.0, "output_tokens": 0, "status": "skipped"}
        )


@dataclass
class PipelineTimer:
    total_rows: int = 0
    total_layers: int = 0
    report_path: str = "timing_report_tpu.json"

    started_at_s: Optional[float] = None
    ended_at_s: Optional[float] = None

    phases: Dict[str, Dict[str, float]] = field(default_factory=dict)
    _phase_starts: Dict[str, float] = field(default_factory=dict)

    rows_done: int = 0
    total_row_time_s: float = 0.0
    total_output_tokens: int = 0
    row_metrics: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.row_timer = RowTimer(parent=self)

    def start(self):
        self.started_at_s = time.perf_counter()

    def start_phase(self, name: str):
        self._phase_starts[name] = time.perf_counter()

    def end_phase(self, name: str):
        t0 = self._phase_starts.get(name)
        if t0 is None:
            return
        dt = time.perf_counter() - t0
        self.phases[name] = {"duration_s": dt}
        self._phase_starts.pop(name, None)

    def _wall_time(self) -> float:
        if self.started_at_s is None:
            return 0.0
        end = self.ended_at_s if self.ended_at_s is not None else time.perf_counter()
        return max(0.0, end - self.started_at_s)

    def finish(self):
        self.ended_at_s = time.perf_counter()
        payload = {
            "total_rows": self.total_rows,
            "total_layers": self.total_layers,
            "rows_done": self.rows_done,
            "wall_time_s": self._wall_time(),
            "avg_row_time_s": (self.total_row_time_s / self.rows_done) if self.rows_done else 0.0,
            "total_output_tokens": self.total_output_tokens,
            "avg_output_tokens_per_row": (self.total_output_tokens / self.rows_done) if self.rows_done else 0.0,
            "tokens_per_second_overall": (self.total_output_tokens / self.total_row_time_s) if self.total_row_time_s > 0 else 0.0,
            "phases": self.phases,
            "rows": self.row_metrics,
            "finished_at_unix": time.time(),
        }

        rp = Path(self.report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n" + "=" * 60)
        print("  TIMING REPORT")
        print("=" * 60)
        print(f"  Rows done:      {self.rows_done}/{self.total_rows}")
        print(f"  Wall time:      {format_duration(payload['wall_time_s'])}")
        print(f"  Avg row time:   {format_duration(payload['avg_row_time_s'])}")
        print(f"  Total tokens:   {self.total_output_tokens:,}")
        print(f"  Throughput:     {payload['tokens_per_second_overall']:.2f} tok/s")
        print(f"  Report path:    {self.report_path}")
        print("=" * 60)
