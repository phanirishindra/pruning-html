# Dataset Processing Pipeline

Downloads 100 rows from [`williambrach/html-description-content`](https://huggingface.co/datasets/williambrach/html-description-content), prunes invisible HTML, applies a 3-task prompt via HuggingFace Inference API, enforces an 8 000 Qwen-token limit per row, and produces two CSV files.

## Output Files

| File | Columns | Description |
|------|---------|-------------|
| `newdataset-1.csv` | `page_id`, `html` | Pruned HTML only |
| `newdataset-2.csv` | `page_id`, `html`, `response` | + model response column |

## Setup

```bash
pip install -r requirements.txt
```

Get a free HuggingFace token at <https://huggingface.co/settings/tokens> (needs Inference API access).

```bash
export HF_API_TOKEN="hf_..."
```

## Run

```bash
python dataset_pipeline.py
```

The script will:

1. Download 100 rows (`page_id`, `html`) and save `newdataset-1.csv`
2. Prune invisible tags (`<style>`, `<script>`, comments, `data-*` attributes)
3. Apply the 3-task prompt to each row via HuggingFace Inference API
4. Enforce 8 000 Qwen tokens per row (40 % HTML / 60 % response split)
5. Save `newdataset-2.csv` with the added `response` column

## Configuration

Edit the constants at the top of `dataset_pipeline.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen2.5-72B-Instruct` | HF Inference model |
| `TOKENIZER_ID` | `Qwen/Qwen2.5-7B-Instruct` | Tokenizer for counting |
| `ROW_COUNT` | `100` | Rows to download |
| `MAX_TOKENS_PER_ROW` | `8000` | Token budget per row |

---

## AirLLM Kaggle Pipeline (NEW)

Run **70B+ parameter models on Kaggle's free GPU** using AirLLM layer-by-layer inference.

### Why AirLLM?

| Feature | HF Inference API | AirLLM (Local) |
|---------|-------------------|------------------|
| GPU needed | None (cloud) | Kaggle T4 (free) |
| Model size | Limited by API | 70B+ params |
| Cost | Free tier limits | Completely free |
| Speed | Fast (cloud GPU) | Slower (layer-by-layer) |
| Privacy | Data sent to HF | 100% local |
| Offline | No | Yes (after download) |

### Model Recommendations

| Rank | Model | Params | Best For |
|------|-------|--------|----------|
| 1st | `Qwen/Qwen2.5-72B-Instruct` | 72B | Verbatim HTML, tables, structured output |
| 2nd | `meta-llama/Llama-3.1-70B-Instruct` | 70B | Strong instruction following |
| 3rd | `mistralai/Mixtral-8x22B-Instruct-v0.1` | 141B | Maximum capacity (MoE) |

### Kaggle Notebook Setup

```python
# Cell 1: Install dependencies
!pip install -q airllm datasets transformers pandas beautifulsoup4 lxml accelerate

# Cell 2: Run the pipeline
!python airllm_kaggle_pipeline.py
```

> **Important:** Enable GPU in Kaggle: Settings > Accelerator > GPU T4 x2

### Local Setup

```bash
pip install -r requirements_airllm.txt
python airllm_kaggle_pipeline.py
```

### Features

- **Layer-by-layer inference**: Only 1 transformer layer in VRAM at a time
- **4-bit compression**: Further reduces memory (configurable)
- **Checkpoint/resume**: Interrupted Kaggle sessions resume where they left off
- **Response validation**: Verifies all 3 tasks are present in output
- **Auto GPU detection**: Falls back to CPU if no GPU available
- **HTML pruning**: Strips invisible tags while preserving structural content
- **Token budget enforcement**: 8000 tokens per row (40% HTML / 60% response)

### Configuration

Edit the `PipelineConfig` dataclass in `airllm_kaggle_pipeline.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `model_id` | `Qwen/Qwen2.5-72B-Instruct` | Model to run via AirLLM |
| `compression` | `4bit` | `4bit`, `8bit`, or `None` |
| `row_count` | `100` | Rows to process |
| `max_tokens_per_row` | `8000` | Token budget per row |
| `max_new_tokens` | `4096` | Max generation length |
| `temperature` | `0.3` | Sampling temperature |
