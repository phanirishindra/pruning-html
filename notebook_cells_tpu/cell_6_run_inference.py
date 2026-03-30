#!/usr/bin/env python3
import os
import gc
import sys
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch_xla.core.xla_model as xm

for _p in ["notebook_cells_tpu", os.path.join(os.getcwd(), "notebook_cells_tpu"), "."]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tpu_config import (
    ROW_COUNT, MODEL_ID, MAX_NEW_TOKENS, MAX_TOKENS_PER_ROW,
    HTML_BUDGET_RATIO, SYSTEM_PROMPT, OUTPUT_FILE_1, CHECKPOINT_FILE,
)
from timing_tracker import format_duration

print("\n" + "=" * 60)
print("  TPU CELL 6: RUN INFERENCE")
print(f"  Model: {MODEL_ID} | Rows: {ROW_COUNT}")
print("=" * 60)

_missing = []
for _var in ["tpu_model", "model_tokenizer", "counting_tokenizer", "pipeline_timer", "tpu_device"]:
    try:
        eval(_var)
    except NameError:
        _missing.append(_var)
if _missing:
    raise RuntimeError(f"Run Cell 5 first. Missing: {_missing}")

if not os.path.exists(OUTPUT_FILE_1):
    raise FileNotFoundError(f"Run Cell 3 first: {OUTPUT_FILE_1}")

df = pd.read_csv(OUTPUT_FILE_1)
df["html"] = df["html"].fillna("")
df["page_id"] = df["page_id"].fillna("")
print(f"  Dataset: {len(df)} rows")

def count_tokens(text):
    return len(counting_tokenizer.encode(text or "", add_special_tokens=False))

def truncate_text(text, max_tok):
    ids = counting_tokenizer.encode(text or "", add_special_tokens=False)
    if len(ids) <= max_tok:
        return text
    return counting_tokenizer.decode(ids[:max_tok], skip_special_tokens=True)

def build_prompt(html):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": html}]
    if hasattr(model_tokenizer, "apply_chat_template"):
        return model_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"System:\n{SYSTEM_PROMPT}\nUser:\n{html}\nAssistant:\n"

def validate_response(resp):
    found = sum(1 for m in ["===TASK 1===", "===TASK 2===", "===TASK 3==="] if m in (resp or ""))
    return {"valid": found == 3 and "USER_QUERY:" in (resp or ""), "tasks_found": found}

def generate_on_tpu(html):
    prompt = build_prompt(html)
    input_ids = model_tokenizer(prompt, return_tensors="pt").input_ids.to(tpu_device)
    try:
        t0 = time.time()
        with torch.no_grad():
            output = tpu_model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )
        xm.mark_step()
        dt = time.time() - t0
        new_ids = output[0][input_ids.shape[1]:]
        token_count = len(new_ids)
        response = model_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        print(f"    {token_count} tokens in {format_duration(dt)} ({(token_count/max(dt,1e-9)):.2f} tok/s)")
        del output, new_ids, input_ids
        gc.collect()
        return response, token_count
    except Exception as e:
        print(f"    ERROR: {e}")
        gc.collect()
        return None, 0

ckpt_path = Path(CHECKPOINT_FILE)
completed_indices, responses = [], {}
if ckpt_path.exists():
    try:
        ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
        completed_indices = ck.get("completed", [])
        responses = {int(k): v for k, v in ck.get("responses", {}).items()}
        print(f"  Checkpoint loaded: {len(completed_indices)} rows")
    except Exception:
        print("  Checkpoint corrupted; starting fresh")

html_budget = int(MAX_TOKENS_PER_ROW * HTML_BUDGET_RATIO)
for idx in df.index:
    if count_tokens(df.at[idx, "html"]) > html_budget:
        df.at[idx, "html"] = truncate_text(df.at[idx, "html"], html_budget)

pipeline_timer.start_phase("generation")
pipeline_timer.row_timer.start_generation_phase()

stats = {"success": 0, "failed": 0, "skipped": len(completed_indices), "valid": 0, "total_tokens": 0}

for idx, row in df.iterrows():
    if idx in completed_indices:
        pipeline_timer.row_timer.skip_row(idx)
        continue

    print(f"\n  Row {idx+1}/{len(df)} | page_id={row['page_id']}")
    pipeline_timer.row_timer.start_row(idx)

    response, out_tok = generate_on_tpu(row["html"])
    if response is None:
        response = "[ERROR] Generation failed."
        stats["failed"] += 1
    else:
        stats["success"] += 1
        stats["total_tokens"] += out_tok
        v = validate_response(response)
        if v["valid"]:
            stats["valid"] += 1
            print("    Validation: PASS")
        else:
            print(f"    Validation: {v['tasks_found']}/3")

    pipeline_timer.row_timer.end_row(idx, output_tokens=out_tok)
    responses[idx] = response
    completed_indices.append(idx)

    if len(completed_indices) % 5 == 0:
        ckpt_path.write_text(json.dumps({
            "completed": completed_indices,
            "responses": {str(k): v for k, v in responses.items()},
            "timestamp": time.time(),
        }, ensure_ascii=False), encoding="utf-8")
        print(f"    Checkpoint saved ({len(completed_indices)} rows)")

pipeline_timer.end_phase("generation")

print("\n" + "=" * 60)
print("  TPU INFERENCE COMPLETE")
print(f"  Success: {stats['success']} | Valid: {stats['valid']} | Failed: {stats['failed']}")
print(f"  Tokens: {stats['total_tokens']:,}")
print("\n  TPU CELL 6 COMPLETE")
print("=" * 60)
