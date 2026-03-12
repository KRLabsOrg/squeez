"""Generate synthetic multi-ecosystem tool output data using an LLM.

Produces canonical v3 JSONL:
{query, background_task, tool_output, gold_spans, tool_type, source}

Architecture:
    Pass 1 — LLM generates query + tool_output inside XML markers (free text, no escaping)
    Pass 2 — LLM picks relevant line numbers (plain text, parsed robustly)
    All samples generated concurrently via asyncio.

Usage:
    python scripts/generate_synthetic_data.py --small-batch --output data/synthetic_small.jsonl
    python scripts/generate_synthetic_data.py --output data/synthetic_train.jsonl
    python scripts/generate_synthetic_data.py --output data/synthetic_train.jsonl --validate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

from squeez.data.canonical import extract_relevant_lines

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "synthetic_tools.yaml"
DEFAULT_OUTPUT_PATH = Path("data/synthetic_train.jsonl")


def load_tool_configs(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)["tool_types"]


# ---------- Prompts -----------------------------------------------------------


def _build_pass1_prompt(
    tool_type: str,
    config: dict,
    scenario: str,
    seed_examples: list[dict],
) -> str:
    examples = ""
    for i, ex in enumerate(seed_examples[:2], 1):
        snippet = "\n".join(ex["tool_output"].split("\n")[:15]) + "\n..."
        example_query = ex.get("query") or ex.get("task", "")
        examples += (
            f"\nExample {i}:\n<query>\n{example_query}\n</query>\n"
            f"<tool_output>\n{snippet}\n</tool_output>\n"
        )

    return f"""Generate a realistic {tool_type} tool output for a coding agent context-pruning task.

Tool type: {tool_type}
Description: {config["description"]}
Scenario: {scenario}
{examples}
Generate a NEW sample. Use exactly this format:

<query>
A short, focused extraction query describing what evidence the agent wants from this one tool output.
</query>
<tool_output>
The realistic raw {tool_type} output (50-300 lines). Include realistic package names, versions, file paths, error messages. Mix relevant and irrelevant output.
</tool_output>"""


def _build_pass2_prompt(tool_type: str, task: str, numbered_output: str) -> str:
    return f"""Given a coding agent's task and {tool_type} tool output, identify which lines are relevant to the task.

TASK: {task}

NUMBERED TOOL OUTPUT:
{numbered_output}

Return a JSON object: {{"lines": [3, 7, 8, 15]}}
Target 5-30% of non-empty lines. Only lines that help diagnose or fix the task."""


# ---------- Parsers -----------------------------------------------------------


def _parse_pass1(text: str) -> tuple[str, str] | None:
    if not text:
        return None
    task_m = re.search(r"<query>\s*\n?(.*?)\n?\s*</query>", text, re.DOTALL)
    out_m = re.search(r"<tool_output>\s*\n?(.*?)\n?\s*</tool_output>", text, re.DOTALL)
    if not task_m or not out_m:
        return None
    task, output = task_m.group(1).strip(), out_m.group(1).strip()
    return (task, output) if task and output else None


def _parse_pass2(text: str) -> list[int] | None:
    if not text:
        return None
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            for key in ("lines", "line_numbers", "relevant_lines"):
                if isinstance(obj.get(key), list):
                    return [int(x) for x in obj[key] if isinstance(x, (int, float))]
        if isinstance(obj, list):
            return [int(x) for x in obj if isinstance(x, (int, float))]
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r"\[[\d,\s\n]+\]", text)
    if m:
        try:
            return [int(x) for x in json.loads(m.group(0))]
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ---------- Async LLM caller --------------------------------------------------


async def _call_llm_async(
    client,
    prompt: str,
    model: str,
    temperature: float = 0.7,
    max_retries: int = 3,
    max_tokens: int = 16384,
) -> str | None:
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
    return None


# ---------- Single sample generation (async) ----------------------------------


async def _generate_one_sample(
    client,
    tool_type: str,
    config: dict,
    model: str,
    scenario: str,
    seed_examples: list[dict],
    temperature: float,
    sample_idx: int,
) -> dict | None:
    """Generate one sample (Pass 1 + Pass 2). Returns sample dict or None."""
    # Pass 1
    p1 = await _call_llm_async(
        client,
        _build_pass1_prompt(tool_type, config, scenario, seed_examples),
        model,
        temperature,
        max_tokens=16384,
    )
    parsed = _parse_pass1(p1)
    if not parsed:
        logger.info(f"  [FAIL] {tool_type} #{sample_idx}: Pass 1 parse failed")
        return None

    query, tool_output = parsed
    output_lines = tool_output.split("\n")
    non_empty = [line for line in output_lines if line.strip()]

    if len(query) < 10:
        logger.info(f"  [FAIL] {tool_type} #{sample_idx}: query too short")
        return None
    if len(non_empty) < 5:
        logger.info(
            f"  [FAIL] {tool_type} #{sample_idx}: output too short ({len(non_empty)} lines)"
        )
        return None

    # Pass 2
    numbered = "\n".join(f"{j + 1}: {ln}" for j, ln in enumerate(output_lines))
    p2 = await _call_llm_async(
        client,
        _build_pass2_prompt(tool_type, query, numbered),
        model,
        temperature=0.0,
        max_tokens=16384,
    )
    line_numbers = _parse_pass2(p2)

    if not line_numbers:
        logger.info(f"  [FAIL] {tool_type} #{sample_idx}: Pass 2 parse failed")
        return None

    # Deduplicate and validate line numbers
    valid_lines: list[int] = []
    seen: set[int] = set()
    for ln in line_numbers:
        if 1 <= ln <= len(output_lines) and ln not in seen:
            if output_lines[ln - 1].strip():
                seen.add(ln)
                valid_lines.append(ln)

    if not valid_lines:
        logger.info(f"  [FAIL] {tool_type} #{sample_idx}: no valid relevant lines")
        return None

    ratio = len(valid_lines) / len(non_empty) if non_empty else 0
    if ratio > 0.6:
        logger.info(f"  [FAIL] {tool_type} #{sample_idx}: too many relevant ({ratio:.0%})")
        return None

    valid_lines.sort()

    # Build spans (consecutive line groups)
    spans = []
    span_start = valid_lines[0]
    prev = valid_lines[0]
    for ln in valid_lines[1:]:
        if ln == prev + 1:
            prev = ln
        else:
            spans.append({"start": span_start, "end": prev, "reason": "relevant"})
            span_start = ln
            prev = ln
    spans.append({"start": span_start, "end": prev, "reason": "relevant"})

    return {
        "instance_id": f"synthetic__{tool_type}-{sample_idx:04d}",
        "tool_type": tool_type,
        "command": f"synthetic {tool_type}",
        "tool_output": tool_output,
        "num_lines": len(output_lines),
        "gold_spans": [
            {"start_line": span["start"], "end_line": span["end"], "reason": "relevant"}
            for span in spans
        ],
        "kept_lines": len(valid_lines),
        "query": query,
        "background_task": "",
        "is_irrelevant": False,
        "source": "synthetic",
    }


# ---------- Batch generation (async) -----------------------------------------


async def generate_all_async(
    config_path: Path,
    output_path: Path,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    small_batch: bool = False,
    validate: bool = False,
    temperature: float = 0.7,
    concurrency: int = 10,
    tool_types: list[str] | None = None,
) -> int:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI()
    tool_configs = load_tool_configs(config_path)

    # Filter to requested tool types
    if tool_types:
        missing = set(tool_types) - set(tool_configs)
        if missing:
            logger.error(f"Unknown tool types: {missing}. Available: {sorted(tool_configs)}")
            return 1
        tool_configs = {k: v for k, v in tool_configs.items() if k in tool_types}

    # Build all tasks upfront
    tasks = []
    for tool_type, config in tool_configs.items():
        target = 5 if small_batch else config["samples"]
        scenarios = config["scenarios"]
        seed_examples = config.get("seed_examples", [])
        for i in range(target):
            scenario = scenarios[i % len(scenarios)]
            tasks.append((tool_type, config, scenario, seed_examples, i))

    logger.info(
        f"Generating {len(tasks)} samples across {len(tool_configs)} tool types (concurrency={concurrency})..."
    )

    # Run with semaphore for concurrency control
    sem = asyncio.Semaphore(concurrency)

    async def bounded(t):
        tool_type, config, scenario, seed_examples, idx = t
        async with sem:
            return await _generate_one_sample(
                client,
                tool_type,
                config,
                model,
                scenario,
                seed_examples,
                temperature,
                idx,
            )

    results = await asyncio.gather(*[bounded(t) for t in tasks])
    all_samples = [r for r in results if r is not None]

    failed = len(tasks) - len(all_samples)
    logger.info(f"\nGenerated {len(all_samples)}/{len(tasks)} samples ({failed} failed)")

    # Generate hard negatives: pair tasks with unrelated tool outputs
    if len(all_samples) >= 2:
        rng = random.Random(42)
        n_negatives = max(1, int(len(all_samples) * 0.2))
        pool = list(all_samples)
        negatives = []
        for _ in range(n_negatives):
            a, b = rng.sample(pool, 2)
            # Take task from a, output from b (different tool type preferred)
            if a["tool_type"] == b["tool_type"]:
                continue
            neg = {
                **b,
                "query": a["query"],
                "background_task": a.get("background_task", ""),
                "instance_id": f"synthetic__neg-{output_path.stem}-{len(negatives):04d}",
                "gold_spans": [],
                "kept_lines": 0,
                "is_irrelevant": True,
                "source": "synthetic_negative",
            }
            negatives.append(neg)
        all_samples.extend(negatives)
        logger.info(f"Added {len(negatives)} hard negative samples (total: {len(all_samples)})")

    # Validation pass
    if validate and all_samples:
        logger.info("\nValidation pass...")
        all_samples, val_report = await _validate_samples_async(client, all_samples, model)
        logger.info(f"After validation: {len(all_samples)} retained")
        with open(output_path.with_suffix(".validation.json"), "w") as f:
            json.dump(val_report, f, indent=2)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    # Stats
    stats: dict[str, dict] = defaultdict(lambda: {"n": 0, "empty": 0, "rel": 0.0})
    for s in all_samples:
        tt = s["tool_type"]
        stats[tt]["n"] += 1
        if not s["gold_spans"]:
            stats[tt]["empty"] += 1
        stats[tt]["rel"] += s.get("kept_lines", 0)

    logger.info(f"\n  {'type':<20s} {'total':>5s} {'empty%':>7s} {'avg_rel':>8s}")
    for tt, d in sorted(stats.items()):
        logger.info(
            f"  {tt:<20s} {d['n']:>5d} {100 * d['empty'] / d['n']:>6.1f}% {d['rel'] / d['n']:>8.1f}"
        )
    logger.info(f"\nWritten to {output_path}")
    return 0


async def _validate_samples_async(
    client,
    samples: list[dict],
    model: str,
    sample_ratio: float = 0.1,
) -> tuple[list[dict], dict]:
    rng = random.Random(42)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_type[s["tool_type"]].append(s)

    validated, report = [], {}
    for tool_type, type_samples in by_type.items():
        n = max(1, int(len(type_samples) * sample_ratio))
        to_check = rng.sample(type_samples, min(n, len(type_samples)))

        agreed = 0
        for s in to_check:
            lines_json = json.dumps(
                extract_relevant_lines(s["tool_output"], s.get("gold_spans", [])),
                indent=2,
            )
            trunc = (
                s["tool_output"][:3000] + "\n..."
                if len(s["tool_output"]) > 3000
                else s["tool_output"]
            )
            prompt = f"""Validate labels for a tool output dataset.

Query: {s["query"]}
Tool output:
{trunc}
Labeled relevant lines:
{lines_json}

Return JSON: {{"agreement_ratio": 0.0-1.0, "notes": "..."}}"""
            resp = await _call_llm_async(client, prompt, model, temperature=0.0)
            try:
                parsed = json.loads(resp.strip()) if resp else None
            except json.JSONDecodeError:
                parsed = None
            if parsed and parsed.get("agreement_ratio", 0) >= 0.7:
                agreed += 1

        rate = agreed / len(to_check) if to_check else 0
        report[tool_type] = {"validated": len(to_check), "agreed": agreed, "rate": rate}
        if rate >= 0.7:
            validated.extend(type_samples)
            logger.info(f"  {tool_type}: PASS ({rate:.0%})")
        else:
            logger.warning(f"  {tool_type}: FAIL ({rate:.0%}) — dropping {len(type_samples)}")

    return validated, report


# ---------- CLI ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate synthetic tool output data")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base-url", default=None)
    p.add_argument("--small-batch", action="store_true")
    p.add_argument(
        "--tool-types",
        nargs="+",
        default=None,
        help="Only generate these tool types (e.g. --tool-types curl grep_output)",
    )
    p.add_argument("--validate", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return asyncio.run(
        generate_all_async(
            config_path=Path(args.config),
            output_path=Path(args.output),
            model=args.model,
            base_url=args.base_url,
            small_batch=args.small_batch,
            validate=args.validate,
            temperature=args.temperature,
            concurrency=args.concurrency,
            tool_types=args.tool_types,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
