#!/usr/bin/env python3
import os
import sys
import gc
import time

for _p in ["notebook_cells_tpu", os.path.join(os.getcwd(), "notebook_cells_tpu"), "."]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tpu_config import (
    MODEL_ID, DTYPE, CACHE_DIR, TOKENIZER_ID,
    ROW_COUNT, TIMING_REPORT_FILE,
)

from timing_tracker import PipelineTimer, detect_layer_count, format_duration

print("\n" + "=" * 60)
print("  TPU CELL 5: LOAD MODEL")
print(f"  Model: {MODEL_ID}")
print(f"  Dtype: {DTYPE}")
print("=" * 60)

# Timing initialization
num_layers = detect_layer_count(MODEL_ID)
pipeline_timer = PipelineTimer(total_rows=ROW_COUNT, total_layers=num_layers, report_path=TIMING_REPORT_FILE)
pipeline_timer.start()

# XLA deterministic/safe env
os.environ.setdefault("XLA_USE_BF16", "1" if DTYPE == "bfloat16" else "0")
os.environ.setdefault("PJRT_DEVICE", "TPU")

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import AutoModelForCausalLM, AutoTokenizer

def _run_worker(index):
    global tpu_model, model_tokenizer, counting_tokenizer
    
    tpu_device = xm.xla_device()
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(DTYPE, torch.bfloat16)

    if xm.is_master_ordinal():
        print(f"\n  [Master] Torch dtype: {torch_dtype}")
        print(f"  [Master] TPU device:  {tpu_device}")
        print(f"\n  Loading model on all cores: {MODEL_ID}")
        pipeline_timer.start_phase("model_loading")

    start = time.time()

    # All workers load the model (sharded by XLA automatically if configured)
    tpu_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Move to TPU core
    tpu_model = tpu_model.to(tpu_device)
    tpu_model.eval()
    xm.mark_step()

    if xm.is_master_ordinal():
        total_load_time = time.time() - start
        print(f"  Total model load & sync: {format_duration(total_load_time)}")
        total_params = sum(p.numel() for p in tpu_model.parameters())
        print(f"  Parameters: {total_params/1e9:.1f}B")

        print(f"\n  Loading tokenizers...")
        model_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
        counting_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
        pipeline_timer.end_phase("model_loading")
        print("  Status: MODEL READY ON ALL CORES")

    xm.rendezvous("model_ready")

# Launcher
print("[launcher] Spawning TPU workers on all available cores (nprocs=None)")
xmp.spawn(_run_worker, args=(), nprocs=None, start_method="fork")

print("\n" + "=" * 60)
print("  TPU CELL 5 COMPLETE")
print("=" * 60)
