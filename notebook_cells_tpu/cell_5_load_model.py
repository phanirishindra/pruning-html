#!/usr/bin/env python3
import os
import sys
import gc
import time
from typing import Optional

# Ensure local imports work in Colab/notebook-export contexts
for _p in ["notebook_cells_tpu", os.path.join(os.getcwd(), "notebook_cells_tpu"), "."]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tpu_config import (
    MODEL_ID,
    DTYPE,
    CACHE_DIR,
    TOKENIZER_ID,
    ROW_COUNT,
    TIMING_REPORT_FILE,
)

from timing_tracker import PipelineTimer, detect_layer_count, format_duration

# -----------------------------------------------------------------------------
# TPU/XLA env defaults (must be set before torch_xla runtime init)
# -----------------------------------------------------------------------------
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1" if DTYPE == "bfloat16" else "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer


def _dtype_from_config(dtype_str: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_str, torch.bfloat16)


def _rank0_print(msg: str):
    # inside worker only
    if xr.global_ordinal() == 0:
        print(msg, flush=True)


def _detect_spawn_procs(default_nprocs: int = 8) -> int:
    """
    Best-effort process count detection for Colab TPU.
    Falls back to 8 (standard v2-8/v3-8/v5e-8 slice layout).
    """
    # Respect manual override first
    env_override = os.environ.get("TPU_NUM_PROCS")
    if env_override:
        try:
            n = int(env_override)
            if n > 0:
                return n
        except ValueError:
            pass

    # Common XLA env hints (if present)
    for k in ["WORLD_SIZE", "TPU_WORLD_SIZE", "XRT_SHARD_WORLD_SIZE"]:
        v = os.environ.get(k)
        if v:
            try:
                n = int(v)
                if n > 0:
                    return n
            except ValueError:
                pass

    return default_nprocs


def _run_worker(index: int):
    device = xm.xla_device()
    rank = xr.global_ordinal()
    world_size = xr.world_size()
    torch_dtype = _dtype_from_config(DTYPE)

    _rank0_print("\n" + "=" * 60)
    _rank0_print("  TPU CELL 5: LOAD MODEL (SHARDED)")
    _rank0_print(f"  Model: {MODEL_ID}")
    _rank0_print(f"  Dtype: {DTYPE} -> {torch_dtype}")
    _rank0_print(f"  World size (TPU cores in use): {world_size}")
    _rank0_print("=" * 60)

    # Hard guard: prevent accidental single-core run for large model
    if world_size < 2:
        if rank == 0:
            raise RuntimeError(
                "Only 1 TPU worker/core detected. This script requires multi-core sharding.\n"
                "Refusing to continue to avoid single-core OOM at model.to(xla:0).\n"
                "Check Colab TPU runtime and run this file as a script via python (not inline-only flow)."
            )
        return

    num_layers = detect_layer_count(MODEL_ID)
    pipeline_timer: Optional[PipelineTimer] = None
    if rank == 0:
        pipeline_timer = PipelineTimer(
            total_rows=ROW_COUNT,
            total_layers=num_layers,
            report_path=TIMING_REPORT_FILE,
        )
        pipeline_timer.start()
        pipeline_timer.start_phase("model_loading")

    _rank0_print(f"\n  Loading model: {MODEL_ID}")
    start = time.time()

    # NOTE: if your transformers version rejects dtype=..., switch to torch_dtype=...
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch_dtype,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    cpu_load = time.time() - start
    _rank0_print(f"  CPU-side model construction complete in {format_duration(cpu_load)}")

    shard_start = time.time()
    model = FSDP(model)
    model = model.to(device)
    model.eval()
    xm.mark_step()
    shard_time = time.time() - shard_start
    _rank0_print(f"  FSDP shard+materialize complete in {format_duration(shard_time)}")

    total_load_time = time.time() - start
    _rank0_print(f"  Total sharded load time: {format_duration(total_load_time)}")

    local_params = sum(p.numel() for p in model.parameters())
    _rank0_print(f"  Local visible params (rank0 view): {local_params/1e9:.2f}B")

    model_tokenizer = None
    counting_tokenizer = None
    if rank == 0:
        _rank0_print(f"\n  Loading tokenizer: {MODEL_ID}")
        tok_start = time.time()
        model_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR
        )
        _rank0_print(f"  Loading counting tokenizer: {TOKENIZER_ID}")
        counting_tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_ID, trust_remote_code=True, cache_dir=CACHE_DIR
        )
        _rank0_print(f"  Tokenizers loaded in {format_duration(time.time() - tok_start)}")

    if rank == 0 and pipeline_timer is not None:
        pipeline_timer.end_phase("model_loading")

    if rank == 0 and model_tokenizer is not None:
        _rank0_print("\n  Running verification...")
        try:
            test_messages = [
                {"role": "system", "content": "Reply with exactly: TPU_OK"},
                {"role": "user", "content": "<p>test</p>"},
            ]
            if hasattr(model_tokenizer, "apply_chat_template"):
                prompt = model_tokenizer.apply_chat_template(
                    test_messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = "Reply with exactly: TPU_OK\nUser: <p>test</p>\nAssistant:"

            input_ids = model_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=20, do_sample=False)

            xm.mark_step()
            res = model_tokenizer.decode(
                out[0][input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            _rank0_print(f"  Output: '{res[:80]}'")
            _rank0_print("  Status: MODEL READY ON TPU (SHARDED)")

            del out, input_ids
            gc.collect()

        except Exception as e:
            _rank0_print(f"  Verification failed (non-fatal): {e}")

    _rank0_print("\n" + "-" * 60)
    _rank0_print("  TPU MODEL READY (SHARDED)")
    _rank0_print(f"  Model:      {MODEL_ID}")
    _rank0_print(f"  Dtype:      {DTYPE}")
    _rank0_print(f"  Device:     {device}")
    _rank0_print(f"  World size: {world_size}")
    _rank0_print(f"  Layers:     {num_layers}")
    _rank0_print(f"  Load time:  {format_duration(total_load_time)}")
    _rank0_print("\n  TPU CELL 5 COMPLETE")
    _rank0_print("=" * 60)


if __name__ == "__main__":
    nprocs = _detect_spawn_procs(default_nprocs=8)
    print(f"[launcher] Spawning TPU workers: nprocs={nprocs}", flush=True)

    # If your runtime complains about "fork", switch to start_method="spawn".
    xmp.spawn(_run_worker, args=(), nprocs=nprocs, start_method="fork")
