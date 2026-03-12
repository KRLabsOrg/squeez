# Training

Squeez supports two model architectures:

- **Generative** (Qwen 3.5 2B + LoRA) — high-quality extraction via XML-wrapped verbatim generation
- **Encoder** (mmBERT 307M) — fast line-level binary classification with sliding window

Both use the same dataset and produce comparable metrics for direct comparison.

## 1. Download the dataset

```bash
python scripts/download_data.py
```

This pulls the released dataset into `data/train.jsonl`, `data/dev.jsonl`, and `data/test.jsonl`.

## 2. Generative model (Qwen + LoRA)

### Known-good environment

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
torch==2.9.0
datasets==3.4.1
```

If training is already working on your machine, do not upgrade these packages casually.

### Train

```bash
squeez train \
    --train-file data/train.jsonl \
    --eval-file data/dev.jsonl
```

### Configuration

Training hyperparameters are in `configs/default.yaml`:

```yaml
model: "Qwen/Qwen3.5-2B"
max_length: 16384
batch_size: 8
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_epochs: 3

lora_r: 16
lora_alpha: 32
lora_dropout: 0
```

Override via CLI:

```bash
squeez train \
    --train-file data/train.jsonl \
    --eval-file data/dev.jsonl \
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

## 3. Encoder model (mmBERT)

The encoder approach is a 307M parameter mmBERT-base with a linear classification head. It classifies each token as relevant (1) or irrelevant (0), then aggregates per line via max-pooling.

### Known-good environment

```bash
pip install -r requirements-encoder.txt
```

### Prepare data

The encoder uses a different input format than the generative model. Convert the canonical/Qwen data:

```bash
python scripts/prepare_encoder_data.py
```

This produces `data/encoder_{train,dev,test}.jsonl` with `{task, tool_output, relevant_lines, tool_type}` derived from canonical spans.

### Train

```bash
python -m squeez.encoder.train \
    --train-file data/encoder_train.jsonl \
    --eval-file data/encoder_dev.jsonl \
    --base-model jhu-clsp/mmBERT-base \
    --output-dir output/squeez_encoder
```

### Configuration

Encoder hyperparameters in `configs/default.yaml`:

```yaml
encoder_base_model: "jhu-clsp/mmBERT-base"
encoder_max_length: 8192
encoder_batch_size: 16
encoder_learning_rate: 2.0e-5
encoder_num_epochs: 5
encoder_warmup_ratio: 0.1
```

Override via CLI:

```bash
python -m squeez.encoder.train \
    --train-file data/encoder_train.jsonl \
    --eval-file data/encoder_dev.jsonl \
    --batch-size 8 \
    --learning-rate 1e-5 \
    --num-epochs 10
```

### Input format

```
[CLS] task_description [SEP] line_1 [LINE_SEP] line_2 [LINE_SEP] ... line_n [SEP]
```

- `[LINE_SEP]` is a special token added to the tokenizer vocabulary
- Task tokens are masked (`label = -100`) during training
- Line tokens receive binary labels: 0 (irrelevant) or 1 (relevant)
- Long samples are split into overlapping sliding windows so every line is supervised

### Sliding window

Both training and inference use sliding windows for tool outputs that exceed the 8K context:

- Lines are split into windows that fit within the token budget
- Windows overlap by 2 lines
- Per-line scores are aggregated via max across windows (if any window says relevant, it's relevant)

## 4. Evaluate

Both models produce the same metrics format for direct comparison:

```bash
# Generative model
squeez eval \
    --extractor-model output/squeez_qwen \
    --eval-file data/test.jsonl

# Encoder model
python -m squeez.encoder.evaluate \
    --model-path output/squeez_encoder \
    --eval-file data/encoder_test.jsonl
```

Metrics computed:

- **Strict line-level F1** — precision/recall against exact ground truth relevant lines
- **Fuzzy line-level F1** — forgiving line overlap at a fixed threshold
- **ROUGE-L** — token-level overlap with reference output
- **Compression ratio** — how much output was filtered
- **Empty accuracy** — correctly predicting empty vs non-empty

Results are saved to `eval_results.json` / `eval_results_encoder.json`.

## 5. Use the trained model

Both model types work through the same API:

```bash
# Either model type — auto-detected
export SQUEEZ_LOCAL_MODEL=./output/squeez_encoder
cat file.py | squeez "Fix the bug"
```

Or in Python:

```python
from squeez.inference.extractor import ToolOutputExtractor

# Generative
extractor = ToolOutputExtractor(model_path="./output/squeez_qwen")

# Encoder (auto-detected from config.json)
extractor = ToolOutputExtractor(model_path="./output/squeez_encoder")

result = extractor.extract(task="Find the relevant evidence block", tool_output=raw)
```
