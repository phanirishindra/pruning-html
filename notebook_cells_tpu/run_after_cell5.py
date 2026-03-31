# --- Colab runner for notebook_cells_tpu/cell_5_load_model.py ---

import os
import subprocess
import textwrap
from pathlib import Path

repo_root = Path("/content/pruning-html")
script_path = repo_root / "notebook_cells_tpu" / "cell_5_load_model.py"

# 1) Sanity checks
print("Repo exists:", repo_root.exists())
print("Script exists:", script_path.exists())
assert repo_root.exists(), "Repo not found at /content/pruning-html. Clone it first."
assert script_path.exists(), f"Script not found: {script_path}"

# 2) TPU env (safe defaults; your script also sets defaults)
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_BF16"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optional: reduce noisy warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# 3) Run as a script (IMPORTANT: not inline cell execution)
cmd = ["python", str(script_path)]
print("Running:", " ".join(cmd))

proc = subprocess.Popen(
    cmd,
    cwd=str(repo_root),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Stream logs live
for line in proc.stdout:
    print(line, end="")

ret = proc.wait()
print(f"\nProcess exit code: {ret}")
if ret != 0:
    raise RuntimeError("cell_5_load_model.py failed. Check logs above.")
