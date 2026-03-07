"""Runtime tool output extractor for agent integration.

Two backends:
- vLLM: connects to a running OpenAI-compatible server (fast, production)
- transformers: loads model locally (no server needed)
"""

import argparse
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


def _load_config() -> dict:
    """Load config from first found config file."""
    for path in CONFIG_SEARCH_PATHS:
        if path.exists():
            import yaml

            with open(path) as f:
                return yaml.safe_load(f) or {}
    return {}


def _format_prompt(task: str, tool_output: str) -> str:
    """Format the input prompt."""
    if len(task) > 3000:
        task = task[:3000] + "..."

    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n"
        f"Task: {task}\n\n"
        f"{tool_output}\n"
        f"<|assistant|>\n"
    )


class ToolOutputExtractor:
    """Extract relevant lines from tool output using a fine-tuned model.

    Supports two backends:
    - vLLM/OpenAI-compatible server: pass base_url
    - Local transformers: pass model_path

    Usage:
        # vLLM (connects to running server)
        extractor = ToolOutputExtractor(base_url="http://localhost:8000/v1")

        # Local
        extractor = ToolOutputExtractor(model_path="./output/qwen-lora")

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

        if not base_url:
            base_url = os.environ.get("SQUEEZ_BASE_URL") or config.get("base_url")
        if not model_path:
            model_path = os.environ.get("SQUEEZ_MODEL_PATH") or config.get("model_path")

        if base_url:
            self._init_vllm(base_url, model_name)
        elif model_path:
            self._init_transformers(model_path, device)
        else:
            raise ValueError(
                "No backend configured. Set model_path or base_url via:\n"
                "  - CLI args: --model-path or --base-url\n"
                "  - Env vars: SQUEEZ_MODEL_PATH or SQUEEZ_BASE_URL\n"
                "  - Config file: squeez.yaml or configs/default.yaml"
            )

    def _init_vllm(self, base_url: str, model_name: str | None):
        """Initialize OpenAI-compatible backend (vLLM, Groq, etc.)."""
        from openai import OpenAI

        api_key = os.environ.get("SQUEEZ_API_KEY") or os.environ.get("OPENAI_API_KEY") or "unused"
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._backend = "vllm"

        # Auto-detect model name from server if not provided
        if not model_name:
            try:
                models = self._client.models.list()
                self._model_name = models.data[0].id
                logger.info(f"Auto-detected model: {self._model_name}")
            except Exception:
                self._model_name = "default"

    def _init_transformers(self, model_path: str, device: str):
        """Initialize local transformers backend."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        if self._backend == "vllm":
            raw = self._extract_vllm(task, tool_output, max_new_tokens, temperature)
        else:
            raw = self._extract_transformers(task, tool_output, max_new_tokens, temperature)

        # Parse JSON response
        try:
            import json
            data = json.loads(raw)
            lines = data.get("relevant_lines", [])
            return "\n".join(lines)
        except (json.JSONDecodeError, TypeError):
            return raw

    def _extract_vllm(
        self, task: str, tool_output: str, max_new_tokens: int, temperature: float
    ) -> str:
        """Extract using OpenAI-compatible server (chat completions API)."""
        if len(task) > 3000:
            task = task[:3000] + "..."

        user_content = f"Task: {task}\n\n{tool_output}" if task else tool_output

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def _extract_transformers(
        self, task: str, tool_output: str, max_new_tokens: int, temperature: float
    ) -> str:
        """Extract using local transformers model."""
        import torch

        prompt = _format_prompt(task, tool_output)

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
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Extract relevant lines from tool output",
        epilog="Reads tool output from stdin or --input-file, writes filtered output to stdout.",
    )
    parser.add_argument("task", nargs="?", default="", help="Task/issue description (optional)")
    parser.add_argument("--input-file", default=None, help="File to read as tool output (default: stdin)")
    parser.add_argument("--model-path", default=None, help="Path to local model (overrides config)")
    parser.add_argument("--base-url", default=None, help="vLLM server URL (overrides config)")
    parser.add_argument("--model-name", default=None, help="Model name on vLLM server (auto-detected if omitted)")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Read input
    if args.input_file:
        with open(args.input_file) as f:
            tool_output = f.read()
    else:
        tool_output = sys.stdin.read()

    extractor = ToolOutputExtractor(
        model_path=args.model_path,
        base_url=args.base_url,
        model_name=args.model_name,
    )

    result = extractor.extract(
        task=args.task,
        tool_output=tool_output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(result)


if __name__ == "__main__":
    main()
