# Dataset

Squeez uses a canonical dataset format and derives model-specific views from it.

Canonical rows contain:

- `query`
- `background_task`
- `tool_output`
- `gold_spans`

Derived views:

- Qwen SFT: `prompt`, `response`, `metadata`
- Encoder: `task`, `tool_output`, `relevant_lines`, `tool_type`

See:

- [Dataset Guide](../guide/dataset.md)
- [Data Generation Guide](../guide/data-generation.md)
