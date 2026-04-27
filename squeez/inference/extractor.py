"""Runtime tool output extractor for agent integration.

Four backends:
- vLLM: connects to a running OpenAI-compatible server (fast, production)
- transformers: loads generative model locally (no server needed)
- encoder: loads discriminative encoder model for token-level line classification
- sentence: loads sentence-level line classifier (per-line with context)
"""

import argparse
import concurrent.futures
import logging
import os
import sys
from pathlib import Path

from squeez.data.config import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

CONFIG_SEARCH_PATHS = [
    Path("squeez.yaml"),
    Path("configs/default.yaml"),
    Path.home() / ".config" / "squeez" / "config.yaml",
]

LOCAL_MODEL_ENV_VARS = ("SQUEEZ_LOCAL_MODEL", "SQUEEZ_MODEL_PATH")
SERVER_URL_ENV_VARS = ("SQUEEZ_SERVER_URL", "SQUEEZ_BASE_URL")
SERVER_MODEL_ENV_VARS = ("SQUEEZ_SERVER_MODEL",)
BACKEND_ENV_VARS = ("SQUEEZ_BACKEND",)


def _load_config() -> dict:
    """Load config from first found config file."""
    for path in CONFIG_SEARCH_PATHS:
        if path.exists():
            import yaml

            with open(path) as f:
                return yaml.safe_load(f) or {}
    return {}


def _first_config_value(config: dict, *keys: str) -> str | None:
    """Return the first configured value from a list of keys."""
    for key in keys:
        value = config.get(key)
        if value:
            return value
    return None


def _first_env_value(*names: str) -> str | None:
    """Return the first non-empty environment variable value."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _build_messages(task: str, tool_output: str) -> list[dict]:
    """Build chat messages for extraction.

    The public API keeps the argument name `task`, but under the v3 benchmark
    this value is the focused extraction query.
    """
    if len(task) > 3000:
        task = task[:3000] + "..."

    user_content = (
        f"<query>\n{task}\n</query>\n<tool_output>\n{tool_output}\n</tool_output>"
        if task
        else f"<tool_output>\n{tool_output}\n</tool_output>"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _detect_local_model_type(model_path: str) -> str:
    """Detect the model backend: 'encoder', 'pooled', 'highlighter', or 'transformers'.

    Handles both local paths and HuggingFace repo ids. Highlighter detection
    looks for an ``auto_map`` entry pointing to a ``*Highlighter`` class — i.e.
    the Verbatim-RAG ModernBERT family (v2, ACL-specialized).
    """
    import json

    # Local path: read config.json directly
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            if config.get("model_type") == "squeez-pooled":
                return "pooled"
            if config.get("model_type") == "squeez-encoder":
                return "encoder"
            auto_map = config.get("auto_map") or {}
            target = auto_map.get("AutoModel") or auto_map.get("AutoModelForTokenClassification")
            if target and "Highlighter" in target:
                return "highlighter"
            return "transformers"
        except (json.JSONDecodeError, OSError):
            return "transformers"

    # Not a local dir — could be a HuggingFace repo id. Probe via AutoConfig.
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        auto_map = getattr(cfg, "auto_map", None) or {}
        target = auto_map.get("AutoModel") or auto_map.get("AutoModelForTokenClassification")
        if target and "Highlighter" in target:
            return "highlighter"
    except Exception:
        pass
    return "transformers"


class ToolOutputExtractor:
    """Extract relevant lines from tool output using a fine-tuned model.

    Supports five backends:
    - vLLM/OpenAI-compatible server: pass base_url
    - Local transformers (generative): pass model_path
    - Encoder (token-level): auto-detected from model config, or backend="encoder"
    - Pooled (line-level): auto-detected from model config, or backend="pooled"
    - Highlighter (Verbatim-RAG ModernBERT family): auto-detected via auto_map,
      or backend="highlighter"

    Usage:
        # vLLM (connects to running server)
        extractor = ToolOutputExtractor(base_url="http://localhost:8000/v1")

        # Local generative
        extractor = ToolOutputExtractor(model_path="./output/qwen-lora")

        # Encoder (auto-detected)
        extractor = ToolOutputExtractor(model_path="./output/squeez_encoder")

        # Pooled line classifier (auto-detected)
        extractor = ToolOutputExtractor(model_path="./output/squeez_pooled")

        # Verbatim-RAG ModernBERT v2 (auto-detected, loads from HuggingFace)
        extractor = ToolOutputExtractor(model_path="KRLabsOrg/verbatim-rag-modern-bert-v2")

        filtered = extractor.extract(task="Fix the bug", tool_output=raw)
    """

    def __init__(
        self,
        model_path: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        max_length: int = 4096,
        device: str = "auto",
    ):
        self.max_length = max_length
        self._backend = None
        self._model = None
        self._tokenizer = None
        self._client = None
        self._model_name = model_name

        # Resolution order: explicit args > env var > config file
        config = _load_config()
        preferred_backend = _first_env_value(*BACKEND_ENV_VARS) or config.get("backend")

        if not model_name:
            model_name = _first_env_value(*SERVER_MODEL_ENV_VARS) or _first_config_value(
                config, "server_model", "model_name"
            )

        if not base_url:
            base_url = _first_env_value(*SERVER_URL_ENV_VARS) or _first_config_value(
                config, "server_url", "base_url"
            )
        if not model_path:
            model_path = _first_env_value(*LOCAL_MODEL_ENV_VARS) or _first_config_value(
                config, "local_model_path", "model_path"
            )

        if preferred_backend == "encoder":
            if not model_path:
                raise ValueError("Backend is set to encoder, but no local model was configured.")
            self._init_encoder(model_path, device)
        elif preferred_backend == "pooled":
            if not model_path:
                raise ValueError("Backend is set to pooled, but no local model was configured.")
            self._init_pooled(model_path, device)
        elif preferred_backend == "highlighter":
            if not model_path:
                raise ValueError(
                    "Backend is set to highlighter, but no local model was configured."
                )
            self._init_highlighter(model_path, device)
        elif preferred_backend == "vllm":
            if not base_url:
                raise ValueError("Backend is set to vllm, but no server URL was configured.")
            self._init_vllm(base_url, model_name)
        elif preferred_backend == "transformers":
            if not model_path:
                raise ValueError(
                    "Backend is set to transformers, but no local model was configured."
                )
            self._init_transformers(model_path, device)
        elif base_url:
            self._init_vllm(base_url, model_name)
        elif model_path:
            # Auto-detect model type from config
            model_type = _detect_local_model_type(model_path)
            if model_type == "pooled":
                self._init_pooled(model_path, device)
            elif model_type == "encoder":
                self._init_encoder(model_path, device)
            elif model_type == "highlighter":
                self._init_highlighter(model_path, device)
            else:
                self._init_transformers(model_path, device)
        else:
            raise ValueError(
                "No backend configured. Set model_path or base_url via:\n"
                "  - CLI args: --local-model or --server-url\n"
                "  - Env vars: SQUEEZ_LOCAL_MODEL or SQUEEZ_SERVER_URL\n"
                "  - Config file: squeez.yaml or configs/default.yaml"
            )

    def _init_pooled(self, model_path: str, device: str):
        """Initialize pooled line classifier backend (single-pass + line-level pool)."""
        import torch
        from transformers import AutoTokenizer

        from squeez.encoder.model import LINE_SEP_TOKEN
        from squeez.encoder.sentence import PooledLineClassifier

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self._tokenizer.convert_tokens_to_ids(LINE_SEP_TOKEN) == self._tokenizer.unk_token_id:
            self._tokenizer.add_special_tokens({"additional_special_tokens": [LINE_SEP_TOKEN]})

        self._model = PooledLineClassifier.from_pretrained(model_path, trust_remote_code=True)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(device)
        self._model.eval()
        self._backend = "pooled"

    def _init_encoder(self, model_path: str, device: str):
        """Initialize encoder-based backend (discriminative line classifier)."""
        import torch
        from transformers import AutoTokenizer

        from squeez.encoder.model import LINE_SEP_TOKEN, SqueezEncoderForLineClassification

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Ensure LINE_SEP is in tokenizer
        if self._tokenizer.convert_tokens_to_ids(LINE_SEP_TOKEN) == self._tokenizer.unk_token_id:
            self._tokenizer.add_special_tokens({"additional_special_tokens": [LINE_SEP_TOKEN]})

        self._model = SqueezEncoderForLineClassification.from_pretrained(
            model_path, trust_remote_code=True
        )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(device)
        self._model.eval()
        self._backend = "encoder"

    def _init_highlighter(self, model_path: str, device: str):
        """Initialize Verbatim-RAG ModernBERT highlighter backend.

        Loads any model that exposes ``.process(question, context, threshold,
        min_span_chars, merge_gap_chars)`` via ``trust_remote_code=True``.
        Examples: ``KRLabsOrg/verbatim-rag-modern-bert-v2``,
        ``KRLabsOrg/acl-verbatim-modernbert``.
        """
        import torch
        from transformers import AutoModel

        self._model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"
        self._model = self._model.to(device)
        self._model.eval()
        self._backend = "highlighter"

        # Recall-tuned defaults — short structured answers are the norm in tool
        # output (file paths, line numbers, error messages). The model card
        # documents these as the recommended config for technical content.
        self._highlighter_threshold = 0.1
        self._highlighter_min_span_chars = 10
        self._highlighter_merge_gap_chars = 20

    def _init_vllm(self, base_url: str, model_name: str | None):
        """Initialize OpenAI-compatible backend (vLLM, Groq, etc.)."""
        from openai import OpenAI

        api_key = os.environ.get("SQUEEZ_API_KEY") or os.environ.get("OPENAI_API_KEY") or "unused"
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._backend = "vllm"

        # Auto-detect model name from server if not provided
        if model_name:
            self._model_name = model_name
        else:
            try:
                models = self._client.models.list()
                self._model_name = models.data[0].id
                logger.info(f"Auto-detected model: {self._model_name}")
            except Exception:
                self._model_name = "default"

    def _init_transformers(self, model_path: str, device: str):
        """Initialize local transformers backend.

        Supports both full models and LoRA/PEFT checkpoints (auto-detected
        via the presence of adapter_config.json).
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_dir = Path(model_path)
        is_lora = (model_dir / "adapter_config.json").exists()

        if is_lora:
            import json as _json

            with open(model_dir / "adapter_config.json") as f:
                adapter_cfg = _json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path", "")
            if not base_model_name:
                raise ValueError(
                    f"LoRA checkpoint at {model_path} has no base_model_name_or_path "
                    f"in adapter_config.json"
                )
            logger.info(f"Loading LoRA checkpoint: base={base_model_name}, adapter={model_path}")
            from peft import PeftModel

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map=device,
                trust_remote_code=True,
            )
            self._model = PeftModel.from_pretrained(base_model, model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map=device,
                trust_remote_code=True,
            )

        self._model.eval()
        self._backend = "transformers"

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def extract(
        self,
        task: str,
        tool_output: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Extract relevant lines from tool output.

        Args:
            task: Description of the coding task/issue
            tool_output: Raw tool output text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Filtered output containing only relevant lines
        """
        if self._backend == "encoder":
            return self._extract_encoder(task, tool_output)

        if self._backend == "pooled":
            return self._extract_pooled(task, tool_output)

        if self._backend == "highlighter":
            return self._extract_highlighter(task, tool_output)

        if self._backend == "vllm":
            raw = self._extract_vllm(task, tool_output, max_new_tokens, temperature)
        else:
            raw = self._extract_transformers(task, tool_output, max_new_tokens, temperature)

        # Parse <relevant_lines> XML response
        import re

        m = re.search(r"<relevant_lines>\s*\n?(.*?)\n?\s*</relevant_lines>", raw, re.DOTALL)
        if m:
            return m.group(1).strip()
        return raw

    def extract_many(
        self,
        items: list[tuple[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        concurrency: int = 1,
    ) -> list[str]:
        """Extract relevant lines for many (task, tool_output) pairs.

        Remote backends can use concurrent requests for higher throughput.
        Local backends fall back to sequential execution.
        """
        if self._backend != "vllm" or concurrency <= 1:
            return [
                self.extract(
                    task=task,
                    tool_output=tool_output,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                for task, tool_output in items
            ]

        results: list[str | None] = [None] * len(items)

        def _run(index: int, item: tuple[str, str]) -> tuple[int, str]:
            task, tool_output = item
            return (
                index,
                self.extract(
                    task=task,
                    tool_output=tool_output,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                ),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_run, i, item) for i, item in enumerate(items)]
            for future in concurrent.futures.as_completed(futures):
                index, result = future.result()
                results[index] = result

        return [result if result is not None else "" for result in results]

    def _extract_pooled(self, task: str, tool_output: str) -> str:
        """Extract using pooled line classifier (single-pass + line-level pool)."""
        lines = self._model.extract(
            task=task,
            tool_output=tool_output,
            tokenizer=self._tokenizer,
        )
        return "\n".join(lines)

    def _extract_encoder(self, task: str, tool_output: str) -> str:
        """Extract using encoder-based line classifier."""
        lines = self._model.extract(
            task=task,
            tool_output=tool_output,
            tokenizer=self._tokenizer,
        )
        return "\n".join(lines)

    def _extract_highlighter(self, task: str, tool_output: str) -> str:
        """Extract using Verbatim-RAG ModernBERT — keep any line touched by a span."""
        if not tool_output:
            return ""
        result = self._model.process(
            question=task,
            context=tool_output,
            threshold=self._highlighter_threshold,
            min_span_chars=self._highlighter_min_span_chars,
            merge_gap_chars=self._highlighter_merge_gap_chars,
        )
        spans = result.get("spans", []) or []
        if not spans:
            return ""
        lines = tool_output.split("\n")
        line_offsets, pos = [], 0
        for line in lines:
            line_offsets.append((pos, pos + len(line)))
            pos += len(line) + 1
        kept_indices: set[int] = set()
        for sp in spans:
            a, b = sp["start"], sp["end"]
            for i, (lo, hi) in enumerate(line_offsets):
                if not (b <= lo or a >= hi):
                    kept_indices.add(i)
        return "\n".join(lines[i] for i in sorted(kept_indices) if lines[i].strip())

    def _extract_vllm(
        self, task: str, tool_output: str, max_new_tokens: int, temperature: float
    ) -> str:
        """Extract using OpenAI-compatible server (chat completions API)."""
        messages = _build_messages(task, tool_output)

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def _extract_transformers(
        self, task: str, tool_output: str, max_new_tokens: int, temperature: float
    ) -> str:
        """Extract using local transformers model."""
        import torch

        messages = _build_messages(task, tool_output)
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return generated.strip()


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Build the parser for the extractor CLI."""
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Extract relevant lines from tool output",
            epilog="Reads tool output from stdin or --input-file, writes filtered output to stdout.",
        )

    parser.add_argument("task", nargs="?", default="", help="Task/issue description (optional)")
    parser.add_argument(
        "--input-file",
        default=None,
        help="File to read as tool output (default: stdin)",
    )
    parser.add_argument(
        "--local-model",
        "--model-path",
        dest="local_model",
        default=None,
        help=(
            "Path or HuggingFace id of an extractor model (overrides config). "
            "Pass `KRLabsOrg/verbatim-rag-modern-bert-v2` to use the published "
            "highlighter model — it will be auto-detected and downloaded."
        ),
    )
    parser.add_argument(
        "--server-url",
        "--base-url",
        dest="server_url",
        default=None,
        help="URL for an OpenAI-compatible model server (overrides config)",
    )
    parser.add_argument(
        "--server-model",
        "--model-name",
        dest="server_model",
        default=None,
        help="Model ID on the remote server (auto-detected if omitted)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print token/line savings to stderr after output",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print accumulated session stats and exit",
    )
    return parser


_STATS_LOG = Path.home() / ".cache" / "squeez" / "session_stats.jsonl"


def _log_stats(in_tokens: int, out_tokens: int, query: str) -> None:
    """Append a stats entry to the session log."""
    import json as _json
    from datetime import datetime, timezone

    _STATS_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "query": query[:80],
    }
    with open(_STATS_LOG, "a") as f:
        f.write(_json.dumps(entry) + "\n")


def _print_summary() -> int:
    """Print accumulated session stats."""
    import json as _json

    if not _STATS_LOG.exists():
        print("No squeez usage recorded yet.")
        return 0

    total_in = 0
    total_out = 0
    calls = 0
    with open(_STATS_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = _json.loads(line)
            total_in += entry["in_tokens"]
            total_out += entry["out_tokens"]
            calls += 1

    saved = total_in - total_out
    saved_pct = (1 - total_out / total_in) * 100 if total_in > 0 else 0

    print(
        f"\033[1m\033[36msqueez session summary\033[0m\n"
        f"\033[90m{'─' * 40}\033[0m\n"
        f"  calls:       {calls:,}\n"
        f"  tokens in:   ~{total_in:,}\n"
        f"  tokens out:  ~{total_out:,}\n"
        f"  \033[32mtokens saved: ~{saved:,} ({saved_pct:.0f}%)\033[0m\n"
        f"\033[90m{'─' * 40}\033[0m"
    )
    return 0


def run(args: argparse.Namespace) -> int:
    """Run the extractor CLI from parsed args."""

    if args.summary:
        return _print_summary()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Read input
    if args.input_file:
        with open(args.input_file) as f:
            tool_output = f.read()
    else:
        tool_output = sys.stdin.read()

    extractor = ToolOutputExtractor(
        model_path=args.local_model,
        base_url=args.server_url,
        model_name=args.server_model,
    )

    result = extractor.extract(
        task=args.task,
        tool_output=tool_output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(result)

    # Compute stats
    in_chars = len(tool_output)
    out_chars = len(result)
    in_tokens = in_chars // 4
    out_tokens = out_chars // 4

    # Always log to session file
    _log_stats(in_tokens, out_tokens, args.task)

    if args.stats:
        in_lines = tool_output.count("\n") + (
            1 if tool_output and not tool_output.endswith("\n") else 0
        )
        out_lines = result.count("\n") + (1 if result and not result.endswith("\n") else 0)
        saved_pct = (1 - out_tokens / in_tokens) * 100 if in_tokens > 0 else 0
        print(
            f"\n\033[90m───\033[0m\n"
            f"\033[90m📥 input:  {in_lines:,} lines  ~{in_tokens:,} tokens\033[0m\n"
            f"\033[90m📤 output: {out_lines:,} lines  ~{out_tokens:,} tokens\033[0m\n"
            f"\033[32m💰 saved:  {saved_pct:.0f}% fewer tokens for your agent\033[0m",
            file=sys.stderr,
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for extraction."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
