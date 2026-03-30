#!/usr/bin/env python3
import os
import sys
import csv
from pathlib import Path

import pandas as pd

for _p in ["notebook_cells_tpu", os.path.join(os.getcwd(), "notebook_cells_tpu"), "."]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tpu_config import (
    OUTPUT_FILE_1, OUTPUT_FILE_2, CHECKPOINT_FILE,
    MAX_TOKENS_PER_ROW, HTML_BUDGET_RATIO, IS_COLAB, IS_KAGGLE,
)

print("\n" + "=" * 60)
print("  TPU CELL 7: SAVE & REPORT")
print("=" * 60)

_missing = []
for _var in ["df", "responses", "pipeline_timer", "counting_tokenizer"]:
    try:
        eval(_var)
    except NameError:
        _missing.append(_var)
if _missing:
    raise RuntimeError(f"Missing: {_missing}. Run Cells 3,5,6 first.")

df["response"] = df.index.map(lambda i: responses.get(i, "[ERROR] No response"))

def count_tokens(text):
    return len(counting_tokenizer.encode(text or "", add_special_tokens=False))

def truncate_text(text, max_tok):
    ids = counting_tokenizer.encode(text or "", add_special_tokens=False)
    if len(ids) <= max_tok:
        return text
    return counting_tokenizer.decode(ids[:max_tok], skip_special_tokens=True)

pipeline_timer.start_phase("token_budget_enforcement")
truncated = 0
for idx in df.index:
    overhead = count_tokens(str(df.at[idx, "page_id"])) + 20
    budget = max(1, MAX_TOKENS_PER_ROW - overhead)
    h_tok = count_tokens(df.at[idx, "html"])
    r_tok = count_tokens(df.at[idx, "response"])
    if h_tok + r_tok > budget:
        h_bud = int(budget * HTML_BUDGET_RATIO)
        r_bud = max(1, budget - h_bud)
        df.at[idx, "html"] = truncate_text(df.at[idx, "html"], h_bud)
        df.at[idx, "response"] = truncate_text(df.at[idx, "response"], r_bud)
        truncated += 1
pipeline_timer.end_phase("token_budget_enforcement")
print(f"  Truncated rows: {truncated}")

pipeline_timer.start_phase("save_dataset_2")
os.makedirs(os.path.dirname(OUTPUT_FILE_2) or ".", exist_ok=True)
df.to_csv(OUTPUT_FILE_2, index=False, quoting=csv.QUOTE_ALL)
pipeline_timer.end_phase("save_dataset_2")

tok_counts = [
    count_tokens(str(r["page_id"])) + count_tokens(r["html"]) + count_tokens(r["response"])
    for _, r in df.iterrows()
]
if tok_counts:
    print(f"  Token stats: min={min(tok_counts):,} max={max(tok_counts):,} mean={sum(tok_counts)//len(tok_counts):,}")
    print(f"  Over budget: {sum(1 for t in tok_counts if t > MAX_TOKENS_PER_ROW)}")

ckpt = Path(CHECKPOINT_FILE)
if ckpt.exists():
    ckpt.unlink()
    print("  Checkpoint removed")

pipeline_timer.finish()

print("\n" + "=" * 60)
print("  FINAL OUTPUT FILES")
print("=" * 60)
for f in [OUTPUT_FILE_1, OUTPUT_FILE_2, "timing_report_tpu.json"]:
    if os.path.exists(f):
        print(f"  {os.path.basename(f):35s} {os.path.getsize(f):>12,} bytes")
    else:
        print(f"  {os.path.basename(f):35s} NOT FOUND")

if IS_COLAB:
    print("\n  Download (Colab):")
    print("    from google.colab import files")
    print(f"    files.download('{OUTPUT_FILE_2}')")
elif IS_KAGGLE:
    print("\n  Download (Kaggle): Save Version > Save & Run All")

print("\n  TPU CELL 7 COMPLETE")
print("=" * 60)
