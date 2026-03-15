# Training & Evaluation Commands

## 1. Download data

```bash
python scripts/download_data.py
```

Downloads from [HuggingFace](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench) to `data/`:
- `train.jsonl`, `dev.jsonl`, `test.jsonl` (generative format)
- `encoder_train.jsonl`, `encoder_dev.jsonl`, `encoder_test.jsonl` (encoder format)
- `canonical_train.jsonl`, `canonical_dev.jsonl`, `canonical_test.jsonl` (span-based ground truth)

## 2. Train

### Pooled encoder (recommended)

Single-pass encoder + line-level mean-pool classifier. Works with any HuggingFace encoder.

```bash
# ModernBERT-base on A100 (~75 min)
python -m squeez.encoder.train \
    --classifier-type pooled \
    --train-file data/encoder_train.jsonl \
    --eval-file data/encoder_dev.jsonl \
    --base-model answerdotai/ModernBERT-base \
    --output-dir output/squeez_pooled \
    --batch-size 96 \
    --gradient-accumulation-steps 2 \
    --max-length 4096 \
    --learning-rate 2e-5 \
    --num-epochs 4

# ModernBERT-large (higher capacity, slower)
python -m squeez.encoder.train \
    --classifier-type pooled \
    --train-file data/encoder_train.jsonl \
    --eval-file data/encoder_dev.jsonl \
    --base-model answerdotai/ModernBERT-large \
    --output-dir output/squeez_pooled_large \
    --batch-size 24 \
    --gradient-accumulation-steps 4 \
    --max-length 4096 \
    --learning-rate 2e-5 \
    --num-epochs 4

# Other encoder models work too
# --base-model jhu-clsp/ettin-encoder-32m
# --base-model microsoft/deberta-v3-large
# --base-model BAAI/bge-large-en-v1.5
```

### Token encoder

Per-token binary classification (alternative approach).

```bash
python -m squeez.encoder.train \
    --classifier-type token \
    --train-file data/encoder_train.jsonl \
    --eval-file data/encoder_dev.jsonl \
    --base-model answerdotai/ModernBERT-base \
    --output-dir output/squeez_encoder \
    --batch-size 2 \
    --max-length 8192
```

### Generative model (Qwen + LoRA)

```bash
squeez train \
    --train-file data/train.jsonl \
    --eval-file data/dev.jsonl \
    --output-dir output/squeez_qwen
```

To merge LoRA weights and serve:

```bash
# Merge
python scripts/merge_lora.py \
    --checkpoint output/squeez_qwen/checkpoint-500 \
    --output output/squeez_qwen_merged

# Serve with vLLM
vllm serve output/squeez_qwen_merged \
    --max-model-len 32768 \
    --trust-remote-code
```

## 3. Evaluate

### Encoder (pooled or token, auto-detected)

```bash
python -m squeez.encoder.evaluate \
    --model-path output/squeez_pooled \
    --eval-file data/encoder_test.jsonl \
    --examples-output eval_examples_pooled.json
```

Optional flags:
- `--threshold 0.5` — relevance probability cutoff (default 0.5)
- `--max-samples 100` — evaluate on a subset

### Generative (local model)

```bash
squeez eval \
    --extractor-model output/squeez_qwen_merged \
    --eval-file data/test.jsonl \
    --max-new-tokens 4096 \
    --examples-output eval_examples.json
```

### Generative (remote vLLM server)

```bash
squeez eval \
    --server-url http://localhost:8000/v1 \
    --eval-file data/test.jsonl \
    --max-new-tokens 4096 \
    --request-concurrency 8 \
    --examples-output eval_examples.json
```

## 4. Standalone inference (no squeez install)

After training the pooled encoder, the output directory contains `modeling_squeez_pooled.py` so `AutoModel` works directly:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("output/squeez_pooled", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("output/squeez_pooled")

result = model.process(
    task="Find the traceback that shows the import error",
    tool_output=open("output.log").read(),
    tokenizer=tokenizer,
    threshold=0.5,
    return_line_probabilities=True,
)
print(result["highlighted_lines"])
print(result["highlighted_indices"])
```

## 5. Upload to HuggingFace

### Dataset

```bash
python scripts/upload_to_hf.py --data-dir data/v3
```

### Model

Push the trained model directory (includes `modeling_squeez_pooled.py` for standalone loading):

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="output/squeez_pooled",
    repo_id="KRLabsOrg/squeez-pooled-modernbert",
    repo_type="model",
)
```
