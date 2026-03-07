# Training

Train your own squeez model using LoRA fine-tuning on Qwen 3.5 2B.

## 1. Download the dataset

```bash
python scripts/download_data.py
```

This pulls the [SWE-bench tool output dataset](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench) into `data/train.jsonl` (7,148 samples) and `data/eval.jsonl` (436 samples).

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
