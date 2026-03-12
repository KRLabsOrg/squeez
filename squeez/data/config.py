"""Configuration for the data generation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

# Tool type weights for generation
TOOL_WEIGHTS = {
    "read_file": 0.28,
    "grep": 0.18,
    "python": 0.08,
    "git_log": 0.08,
    "test_output": 0.08,
    "git_diff": 0.05,
    "git_blame": 0.04,
    "ls": 0.04,
    "lint_output": 0.02,
    "build_output": 0.02,
    "curl": 0.03,
    "pip_install": 0.04,
    "type_check": 0.04,
    "coverage": 0.02,
}

# Quality filter thresholds
MIN_RELEVANT_RATIO = 0.02
MAX_RELEVANT_RATIO = 0.40
MIN_RELEVANT_LINES = 3
MIN_TOTAL_LINES = 10
MAX_TOOL_OUTPUT_LINES = 500
MAX_TOOL_PROMPT_LINE_CHARS = 400
MAX_TOOL_PROMPT_TOTAL_CHARS = 60000

# System prompt for the extraction model
SYSTEM_PROMPT = (
    "You prune verbose tool output for a coding agent. "
    "Given a focused extraction query and one tool output, return only the "
    "smallest verbatim evidence block(s) the agent should read next. "
    "Return the kept text inside <relevant_lines> tags. "
    "Do not rewrite, summarize, or invent lines."
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
    generate_queries_with_teacher: bool = True

    # Execution
    command_timeout: int = 30  # seconds per command

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.source_cache_dir = Path(self.source_cache_dir)
        self.repos_dir = Path(self.repos_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source_cache_dir.mkdir(parents=True, exist_ok=True)
        self.repos_dir.mkdir(parents=True, exist_ok=True)
