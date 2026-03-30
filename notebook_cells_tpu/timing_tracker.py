from __future__ import annotations

import json
import os
import socket
import time
import uuid
import threading
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TimingEvent:
    run_id: str
    tracker_name: str
    step: str
    parent_step: Optional[str]
    depth: int
    start_ts: float
    end_ts: float
    duration_s: float
    meta: Dict[str, Any]


class TimingTracker:
    """
    Production timing tracker for notebook/TPU workflows.

    Features:
    - Global run metadata (run_id, repo, platform, device)
    - Hierarchical nested timing spans
    - Inference metric logging (TTFT, tokens/s, prompt/completion/total tokens)
    - Persistent JSONL event logs under artifacts/timing/
    - Summary tables and optional pandas DataFrame export
    """

    def __init__(
        self,
        tracker_name: str = "global",
        repo_slug: Optional[str] = None,
        platform_name: Optional[str] = None,
        device_name: Optional[str] = None,
        persist: bool = True,
        artifacts_dir: str = "artifacts/timing",
        auto_print: bool = True,
    ):
        self.tracker_name = tracker_name
        self.repo_slug = repo_slug or os.getenv("REPO_SLUG", "unknown/unknown")
        self.platform_name = platform_name or self._detect_platform()
        self.device_name = device_name or self._detect_device()
        self.persist = persist
        self.auto_print = auto_print

        self.run_id = self._new_run_id()
        self.run_started_at = time.perf_counter()
        self.run_started_at_iso = datetime.now(timezone.utc).isoformat()

        self._lock = threading.RLock()
        self._events: List[TimingEvent] = []
        self._stack: List[Dict[str, Any]] = []  # active nested spans

        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self.artifacts_dir / f"{self.run_id}.jsonl"

        self._write_run_header()

    # ----------------------------
    # Core run/session utilities
    # ----------------------------
    @staticmethod
    def _new_run_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"run_{ts}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _detect_platform() -> str:
        if os.path.exists("/content") and not os.path.exists("/kaggle"):
            return "colab"
        if os.path.exists("/kaggle/working"):
            return "kaggle"
        return "local"

    @staticmethod
    def _detect_device() -> str:
        # Lightweight detection, avoids hard dependency on torch/torch_xla
        if os.getenv("COLAB_TPU_ADDR") or os.getenv("TPU_NAME"):
            return "tpu"
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                return f"gpu:{torch.cuda.get_device_name(0)}"
        except Exception:
            pass
        return "cpu"

    def _base_meta(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tracker_name": self.tracker_name,
            "repo_slug": self.repo_slug,
            "platform": self.platform_name,
            "device": self.device_name,
            "host": socket.gethostname(),
            "started_at_utc": self.run_started_at_iso,
        }

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        if not self.persist:
            return
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_run_header(self) -> None:
        self._write_jsonl({
            "type": "run_header",
            **self._base_meta(),
        })

    # ----------------------------
    # Hierarchical timing spans
    # ----------------------------
    def start(self, step: str, meta: Optional[Dict[str, Any]] = None) -> None:
        now = time.perf_counter()
        with self._lock:
            parent = self._stack[-1]["step"] if self._stack else None
            depth = len(self._stack)
            self._stack.append({
                "step": step,
                "start": now,
                "parent": parent,
                "depth": depth,
                "meta": meta or {},
            })

    def stop(self, step: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> float:
        now = time.perf_counter()
        with self._lock:
            if not self._stack:
                raise RuntimeError("No active timing span to stop.")

            active = self._stack.pop()

            # Optional safety check
            if step is not None and active["step"] != step:
                raise RuntimeError(
                    f"Timing stop mismatch: expected '{active['step']}', got '{step}'."
                )

            duration = now - active["start"]
            merged_meta = {}
            merged_meta.update(self._base_meta())
            merged_meta.update(active.get("meta", {}))
            if meta:
                merged_meta.update(meta)

            event = TimingEvent(
                run_id=self.run_id,
                tracker_name=self.tracker_name,
                step=active["step"],
                parent_step=active["parent"],
                depth=active["depth"],
                start_ts=active["start"],
                end_ts=now,
                duration_s=duration,
                meta=merged_meta,
            )
            self._events.append(event)

            self._write_jsonl({
                "type": "timing_event",
                **asdict(event),
            })

        if self.auto_print:
            indent = "  " * event.depth
            print(f"[timing] {indent}{event.step}: {event.duration_s:.3f}s")
        return duration

    @contextmanager
    def track(self, step: str, meta: Optional[Dict[str, Any]] = None):
        self.start(step, meta=meta)
        try:
            yield
        finally:
            self.stop(step=step)

    # ----------------------------
    # LLM-specific metrics
    # ----------------------------
    def log_inference_metrics(
        self,
        step: str,
        ttft_s: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        wall_time_s: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Logs inference quality/speed metrics as an event-like record.
        tokens_per_s = completion_tokens / (wall_time_s - ttft_s) when possible.
        """
        calc_total = total_tokens
        if calc_total is None and prompt_tokens is not None and completion_tokens is not None:
            calc_total = prompt_tokens + completion_tokens

        gen_time = None
        tps = None
        if wall_time_s is not None and ttft_s is not None:
            gen_time = max(wall_time_s - ttft_s, 1e-9)
            if completion_tokens is not None:
                tps = completion_tokens / gen_time
        elif wall_time_s is not None and completion_tokens is not None:
            tps = completion_tokens / max(wall_time_s, 1e-9)

        payload = {
            "type": "inference_metrics",
            **self._base_meta(),
            "step": step,
            "ttft_s": ttft_s,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": calc_total,
            "wall_time_s": wall_time_s,
            "generation_time_s": gen_time,
            "tokens_per_s": tps,
        }
        if meta:
            payload["meta"] = meta

        self._write_jsonl(payload)
        if self.auto_print:
            print(
                f"[metrics] {step} | ttft={ttft_s} | wall={wall_time_s} | "
                f"prompt={prompt_tokens} completion={completion_tokens} total={calc_total} | "
                f"tps={None if tps is None else round(tps, 3)}"
            )

    # ----------------------------
    # Summaries / exports
    # ----------------------------
    def events(self) -> List[TimingEvent]:
        with self._lock:
            return list(self._events)

    def total_wall_s(self) -> float:
        return time.perf_counter() - self.run_started_at

    def total_timed_s(self) -> float:
        with self._lock:
            return sum(e.duration_s for e in self._events)

    def summary_text(self, sort_desc: bool = True) -> str:
        evs = self.events()
        if sort_desc:
            evs = sorted(evs, key=lambda e: e.duration_s, reverse=True)

        total = sum(e.duration_s for e in evs) or 1e-9

        lines = []
        lines.append(f"Timing Summary [{self.tracker_name}] run_id={self.run_id}")
        lines.append("-" * 96)
        lines.append(f"{'Step':45} {'Depth':>5} {'Seconds':>12} {'Pct':>8} {'Parent':20}")
        lines.append("-" * 96)
        for e in evs:
            pct = (e.duration_s / total) * 100.0
            lines.append(
                f"{e.step[:45]:45} {e.depth:5d} {e.duration_s:12.3f} {pct:7.2f}% "
                f"{(e.parent_step or '-')[:20]:20}"
            )
        lines.append("-" * 96)
        lines.append(f"{'TOTAL_TIMED':45} {'':5} {self.total_timed_s():12.3f} {100.00:7.2f}% {'-':20}")
        lines.append(f"{'TOTAL_WALL':45} {'':5} {self.total_wall_s():12.3f} {'-':>8} {'-':20}")
        return "\n".join(lines)

    def print_summary(self, sort_desc: bool = True) -> None:
        print(self.summary_text(sort_desc=sort_desc))

    def to_json(self, path: str) -> None:
        payload = {
            "run_id": self.run_id,
            "tracker_name": self.tracker_name,
            "base_meta": self._base_meta(),
            "total_wall_s": self.total_wall_s(),
            "total_timed_s": self.total_timed_s(),
            "events": [asdict(e) for e in self.events()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def to_pandas(self):
        import pandas as pd  # optional dependency
        rows = []
        for e in self.events():
            row = asdict(e)
            # flatten meta selectively
            for k, v in (e.meta or {}).items():
                row[f"meta.{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)


# ----------------------------
# Global singleton helpers
# ----------------------------
_DEFAULT_TRACKER: Optional[TimingTracker] = None

def configure_tracker(
    tracker_name: str = "global",
    repo_slug: Optional[str] = None,
    platform_name: Optional[str] = None,
    device_name: Optional[str] = None,
    persist: bool = True,
    artifacts_dir: str = "artifacts/timing",
    auto_print: bool = True,
) -> TimingTracker:
    global _DEFAULT_TRACKER
    _DEFAULT_TRACKER = TimingTracker(
        tracker_name=tracker_name,
        repo_slug=repo_slug,
        platform_name=platform_name,
        device_name=device_name,
        persist=persist,
        artifacts_dir=artifacts_dir,
        auto_print=auto_print,
    )
    return _DEFAULT_TRACKER

def get_tracker() -> TimingTracker:
    global _DEFAULT_TRACKER
    if _DEFAULT_TRACKER is None:
        _DEFAULT_TRACKER = TimingTracker()
    return _DEFAULT_TRACKER

@contextmanager
def track(step: str, meta: Optional[Dict[str, Any]] = None):
    t = get_tracker()
    with t.track(step, meta=meta):
        yield
