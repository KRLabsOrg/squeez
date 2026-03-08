# Training

Train your own squeez model using LoRA fine-tuning on Qwen 3.5 2B.

## 1. Download the dataset

```bash
python scripts/download_data.py
```

This pulls the [SWE-bench tool output dataset](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench) into `data/train.jsonl` (7,148 samples) and `data/eval.jsonl` (436 samples).

## Known-good environment

This repo currently has a known-good training stack pinned in [requirements-train.txt](/Users/adamkovacs/projects/squeez/requirements-train.txt).

Install it with:

```bash
pip install -r requirements-train.txt
```

Pinned versions:

```txt
unsloth==2026.3.4
unsloth_zoo==2026.3.2
trl==0.24.0
transformers==5.2.0
peft==0.18.1
torch==2.10.0
datasets==3.4.1
```

If training is already working on your machine, do not upgrade these packages casually.

## 2. Train

```bash
squeez train \
    --train-file data/train.jsonl \
    --eval-file data/eval.jsonl
```

### Configuration

Training hyperparameters are in `configs/default.yaml`:

```yaml
model: "Qwen/Qwen3.5-2B"
max_length: 32768
batch_size: 2
gradient_accumulation_steps: 8  # effective batch size: 16
learning_rate: 2.0e-4
num_epochs: 3

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
```

Override via CLI:

```bash
squeez train \
    --train-file data/train.jsonl \
    --eval-file data/eval.jsonl \
    --base-model Qwen/Qwen3.5-2B \
    --batch-size 4 \
    --lr 1e-4 \
    --epochs 5 \
    --lora-r 32
```

### LoRA targets

LoRA adapters are applied to all attention and FFN layers:

- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention)
- `gate_proj`, `up_proj`, `down_proj` (FFN)

With r=16, this trains ~0.5% of total parameters.

## 3. Evaluate

```bash
squeez eval \
    --extractor-model output/squeez_qwen \
    --eval-file data/eval.jsonl
```

Metrics computed:

- **Line-level F1** — precision/recall against ground truth relevant lines
- **ROUGE-L** — token-level overlap with reference output
- **Compression ratio** — how much output was filtered

Results are saved to `eval_results.json`.

## 4. Use the trained model

```bash
export SQUEEZ_LOCAL_MODEL=./output/squeez_qwen
cat file.py | squeez "Fix the bug"
```

Or in Python:

```python
from squeez.inference.extractor import ToolOutputExtractor

extractor = ToolOutputExtractor(model_path="./output/squeez_qwen")
result = extractor.extract(task="Fix the bug", tool_output=raw)
```
