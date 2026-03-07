"""Configuration for the data generation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

# Tool type weights for generation
TOOL_WEIGHTS = {
    "read_file": 0.25,
    "grep": 0.15,
    "python": 0.10,
    "git_log": 0.10,
    "test_output": 0.10,
    "git_diff": 0.05,
    "git_blame": 0.05,
    "ls": 0.05,
    "lint_output": 0.05,
    "build_output": 0.05,
    "curl": 0.05,
}

# Quality filter thresholds
MIN_RELEVANT_RATIO = 0.02
MAX_RELEVANT_RATIO = 0.40
MIN_RELEVANT_LINES = 3
MIN_TOTAL_LINES = 10
MAX_TOOL_OUTPUT_LINES = 500

# System prompt for the extraction model
SYSTEM_PROMPT = (
    "You extract relevant lines from tool output for a coding task. "
    'Return a JSON object: {"relevant_lines": ["line1", "line2", ...]}. '
    "Include ONLY lines the agent needs to see."
)


@dataclass
class PipelineConfig:
    """Configuration for the data generation pipeline."""

    # Paths
    output_dir: Path = Path("data")
    source_cache_dir: Path = Path("data/source_cache")
    repos_dir: Path = Path("data/repos")

    # API
    github_token: str = ""
    openai_api_key: str = ""
    distillation_model: str = "gpt-5.4"
    distillation_base_url: str | None = None  # Custom API base URL (e.g. Groq)

    # SWE-bench
    swebench_dataset: str = "princeton-nlp/SWE-bench"
    splits: list[str] = field(default_factory=lambda: ["test"])
    max_instances: int | None = None

    # Tool generation
    min_tools_per_instance: int = 3
    max_tools_per_instance: int = 7

    # Output limits
    max_tool_output_lines: int = MAX_TOOL_OUTPUT_LINES

    # Distillation
    distillation_max_concurrent: int = 50
    distillation_temperature: float = 0.3

    # Execution
    command_timeout: int = 30  # seconds per command

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.source_cache_dir = Path(self.source_cache_dir)
        self.repos_dir = Path(self.repos_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source_cache_dir.mkdir(parents=True, exist_ok=True)
        self.repos_dir.mkdir(parents=True, exist_ok=True)
