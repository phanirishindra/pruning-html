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

# Timing
num_layers = detect_layer_count(MODEL_ID)
pipeline_timer = PipelineTimer(total_rows=ROW_COUNT, total_layers=num_layers, report_path=TIMING_REPORT_FILE)
pipeline_timer.start()
pipeline_timer.start_phase("model_loading")

# XLA deterministic/safe env
os.environ.setdefault("XLA_USE_BF16", "1" if DTYPE == "bfloat16" else "0")
os.environ.setdefault("PJRT_DEVICE", "TPU")

import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer

tpu_device = xm.xla_device()

dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
torch_dtype = dtype_map.get(DTYPE, torch.bfloat16)

print(f"\n  Torch dtype: {torch_dtype}")
print(f"  TPU device:  {tpu_device}")

# rough memory check (pre-flight)
model_hint_gb = (
    64 if "32B" in MODEL_ID else
    144 if "72B" in MODEL_ID else
    28 if "14B" in MODEL_ID else
    14 if "7B" in MODEL_ID else
    0
)
if model_hint_gb:
    print(f"  Model memory hint (bf16/f16): ~{model_hint_gb} GB total")

if "32B" in MODEL_ID and os.environ.get("PJRT_DEVICE", "TPU") == "TPU":
    print("  WARNING: 32B bf16 on free Colab TPU v2-8 is very likely to OOM. Prefer a 14B model.")

print(f"\n  Loading model: {MODEL_ID}")
start = time.time()

tpu_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

cpu_load = time.time() - start
print(f"  CPU load complete in {format_duration(cpu_load)}")

move_start = time.time()
tpu_model = tpu_model.to(tpu_device)
tpu_model.eval()
xm.mark_step()
move_time = time.time() - move_start
print(f"  TPU transfer complete in {format_duration(move_time)}")

total_load_time = time.time() - start
print(f"  Total model load: {format_duration(total_load_time)}")

total_params = sum(p.numel() for p in tpu_model.parameters())
print(f"  Parameters: {total_params/1e9:.1f}B")

print(f"\n  Loading tokenizer: {MODEL_ID}")
tok_start = time.time()
model_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
print(f"  Loading counting tokenizer: {TOKENIZER_ID}")
counting_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
print(f"  Tokenizers loaded in {format_duration(time.time()-tok_start)}")

pipeline_timer.end_phase("model_loading")

print("\n  Running verification...")
try:
    test_messages = [
        {"role": "system", "content": "Reply with exactly: TPU_OK"},
        {"role": "user", "content": "<p>test</p>"},
    ]
    if hasattr(model_tokenizer, "apply_chat_template"):
        prompt = model_tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "Reply with exactly: TPU_OK\nUser: <p>test</p>\nAssistant:"

    input_ids = model_tokenizer(prompt, return_tensors="pt").input_ids.to(tpu_device)

    with torch.no_grad():
        out = tpu_model.generate(input_ids, max_new_tokens=20, do_sample=False)

    xm.mark_step()
    res = model_tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"  Output: '{res[:80]}'")
    print("  Status: MODEL READY ON TPU")

    del out, input_ids
    gc.collect()

except Exception as e:
    print(f"  Verification failed (non-fatal): {e}")

print("\n" + "-" * 60)
print("  TPU MODEL READY")
print(f"  Model:      {MODEL_ID}")
print(f"  Params:     {total_params/1e9:.1f}B")
print(f"  Dtype:      {DTYPE}")
print(f"  Device:     {tpu_device}")
print(f"  Layers:     {num_layers}")
print(f"  Load time:  {format_duration(total_load_time)}")
print("\n  TPU CELL 5 COMPLETE")
print("=" * 60)
