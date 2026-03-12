"""Relabel existing raw tool outputs with focused queries and canonical spans.

Supports:
- SWE raw tool outputs with background task recovered from an existing dataset
- synthetic raw tool outputs with background task from the row itself
- optional generation of synthetic negatives by mismatching queries and outputs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path

from openai import AsyncOpenAI

from squeez.data.canonical import canonical_record
from squeez.data.config import (
    MAX_TOOL_OUTPUT_LINES,
    MAX_TOOL_PROMPT_LINE_CHARS,
    MAX_TOOL_PROMPT_TOTAL_CHARS,
)
from squeez.data.llm_distiller import (
    CONTENT_FIRST_QUERY_PROMPT,
    DISTILLATION_PROMPT,
    QUERY_PROMPT,
    _fallback_query,
    _parse_query_response,
    _parse_spans_response,
    _query_is_compatible,
    _tool_guidance,
)

logger = logging.getLogger(__name__)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None


def _key(row: dict) -> str:
    return f"{row.get('instance_id', '')}::{row.get('command', '')}"


def _number_output(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))


def _number_output_for_prompt(
    text: str,
    *,
    max_lines: int = MAX_TOOL_OUTPUT_LINES,
    max_chars_per_line: int = MAX_TOOL_PROMPT_LINE_CHARS,
    max_total_chars: int = MAX_TOOL_PROMPT_TOTAL_CHARS,
) -> str:
    """Build a numbered teacher view with bounded line/character budgets.

    This is only for teacher prompting. The canonical labels still refer back to
    the original raw output lines.
    """
    lines = text.split("\n")
    bounded = lines[:max_lines]
    rendered = []
    total_chars = 0
    rendered_source_lines = 0
    for i, line in enumerate(bounded, start=1):
        if len(line) > max_chars_per_line:
            line = line[:max_chars_per_line] + " ... [truncated for prompt]"
        numbered = f"{i}: {line}"
        next_total = total_chars + len(numbered) + 1
        if next_total > max_total_chars:
            rendered.append(f"{i}: ... [prompt view truncated at {max_total_chars} chars]")
            break
        rendered.append(numbered)
        total_chars = next_total
        rendered_source_lines = i
    omitted_lines = max(0, len(lines) - rendered_source_lines)
    if omitted_lines > 0:
        rendered.append(
            f"{rendered_source_lines + 1}: ... [{omitted_lines} lines omitted for prompt]"
        )
    return "\n".join(rendered)


def _load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in rows:
        key = _key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _load_task_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    rows = _load_rows(path)
    mapping: dict[str, str] = {}
    for row in rows:
        task = row.get("background_task") or row.get("task")
        if task:
            mapping[_key(row)] = str(task)
    return mapping


def _partial_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.partial")


async def _call_query(
    client: AsyncOpenAI,
    model: str,
    *,
    background_task: str,
    tool_type: str,
    command: str,
    raw_output: str,
    query_hint: str | None = None,
) -> str | None:
    prompt = QUERY_PROMPT.format(
        issue_text=background_task or "(none)",
        tool_type=tool_type,
        command=command,
        numbered_output=_number_output_for_prompt(raw_output),
        tool_guidance=_tool_guidance(tool_type),
        query_hint=query_hint or "(none)",
    )
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You write short evidence-extraction queries for coding-agent tool outputs. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw_content = response.choices[0].message.content
            if raw_content is None:
                logger.warning("Query generation returned empty content on attempt %d", attempt + 1)
                continue
            raw = raw_content.strip()
            parsed = _parse_query_response(raw)
            if parsed and _query_is_compatible(tool_type, parsed):
                return parsed
            if parsed:
                logger.warning(
                    "Query generation returned incompatible query on attempt %d for %s: %r",
                    attempt + 1,
                    tool_type,
                    parsed[:200],
                )
            logger.warning(
                "Query generation returned unparsable JSON on attempt %d: %r",
                attempt + 1,
                raw[:200],
            )
        except Exception as exc:
            logger.warning("Query generation failed on attempt %d: %s", attempt + 1, exc)
    return _fallback_query(tool_type, query_hint)


async def _call_content_first_query(
    client: AsyncOpenAI,
    model: str,
    *,
    background_task: str,
    tool_type: str,
    command: str,
    raw_output: str,
    query_hint: str | None = None,
) -> str:
    prompt = CONTENT_FIRST_QUERY_PROMPT.format(
        issue_text=background_task or "(none)",
        tool_type=tool_type,
        command=command,
        numbered_output=_number_output_for_prompt(raw_output),
    )
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You write short evidence-extraction queries for coding-agent tool outputs. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw_content = response.choices[0].message.content
            if raw_content is None:
                continue
            raw = raw_content.strip()
            parsed = _parse_query_response(raw)
            if parsed and _query_is_compatible(tool_type, parsed):
                return parsed
        except Exception as exc:
            logger.warning(
                "Content-first query generation failed on attempt %d: %s", attempt + 1, exc
            )
    return _fallback_query(tool_type, query_hint)


async def _call_spans(
    client: AsyncOpenAI,
    model: str,
    *,
    background_task: str,
    query: str,
    tool_type: str,
    command: str,
    raw_output: str,
) -> list[dict] | None:
    prompt = DISTILLATION_PROMPT.format(
        issue_text=background_task or "(none)",
        query=query,
        tool_type=tool_type,
        command=command,
        total_lines=len(raw_output.split("\n")),
        numbered_output=_number_output_for_prompt(raw_output),
    )
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You select contiguous evidence spans from tool output. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=8192,
            )
            raw_content = response.choices[0].message.content
            if raw_content is None:
                logger.warning("Span generation returned empty content on attempt %d", attempt + 1)
                continue
            raw = raw_content.strip()
            spans, _summary = _parse_spans_response(raw)
            if spans is not None:
                return spans
            logger.warning(
                "Span generation returned unparsable JSON on attempt %d: %r", attempt + 1, raw[:200]
            )
        except Exception as exc:
            logger.warning("Span generation failed on attempt %d: %s", attempt + 1, exc)
    return None


async def _relabel_one(
    client: AsyncOpenAI,
    model: str,
    row: dict,
    task_map: dict[str, str],
    default_source: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    async with semaphore:
        raw_output = str(row.get("tool_output") or row.get("output") or "")
        if not raw_output.strip():
            return None

        background_task = str(
            row.get("background_task") or row.get("task") or task_map.get(_key(row), "")
        )
        tool_type = str(row["tool_type"])
        command = str(row.get("command", ""))
        query_hint = row.get("query")

        query = await _call_query(
            client,
            model,
            background_task=background_task,
            tool_type=tool_type,
            command=command,
            raw_output=raw_output,
            query_hint=query_hint,
        )
        if not query:
            return None

        spans = await _call_spans(
            client,
            model,
            background_task=background_task,
            query=query,
            tool_type=tool_type,
            command=command,
            raw_output=raw_output,
        )
        if spans is None:
            return None
        if not spans and str(row.get("source") or default_source) != "synthetic_negative":
            content_first_query = await _call_content_first_query(
                client,
                model,
                background_task=background_task,
                tool_type=tool_type,
                command=command,
                raw_output=raw_output,
                query_hint=query_hint,
            )
            if content_first_query and content_first_query != query:
                query = content_first_query
                spans = await _call_spans(
                    client,
                    model,
                    background_task=background_task,
                    query=query,
                    tool_type=tool_type,
                    command=command,
                    raw_output=raw_output,
                )
            if not spans:
                return None

        return canonical_record(
            instance_id=str(row["instance_id"]),
            source=str(row.get("source") or default_source),
            tool_type=tool_type,
            query=query,
            background_task=background_task,
            tool_output=raw_output,
            gold_spans=spans,
            command=command,
        )


def _add_negatives(rows: list[dict], ratio: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    positives = [row for row in rows if row.get("source") != "synthetic_negative"]
    if len(positives) < 2 or ratio <= 0:
        return rows

    n_neg = max(1, int(len(positives) * ratio))
    negatives: list[dict] = []
    attempts = 0
    while len(negatives) < n_neg and attempts < n_neg * 10:
        attempts += 1
        query_row, output_row = rng.sample(positives, 2)
        if query_row["tool_type"] == output_row["tool_type"]:
            continue
        negatives.append(
            canonical_record(
                instance_id=f"synthetic__neg-relabel-{len(negatives):04d}",
                source="synthetic_negative",
                tool_type=output_row["tool_type"],
                query=query_row["query"],
                background_task=query_row.get("background_task", ""),
                tool_output=output_row["tool_output"],
                gold_spans=[],
                command=output_row.get("command", ""),
                is_irrelevant=True,
            )
        )
        negatives[-1]["split_group_id"] = output_row.get(
            "split_group_id", output_row["instance_id"]
        )
        negatives[-1]["query_origin_instance_id"] = query_row["instance_id"]
        negatives[-1]["output_origin_instance_id"] = output_row["instance_id"]
    return rows + negatives


async def relabel_async(
    *,
    input_path: Path,
    output_path: Path,
    model: str,
    base_url: str | None,
    task_source: Path | None,
    default_source: str,
    concurrency: int,
    add_negatives: bool,
    negative_ratio: float,
    seed: int,
) -> int:
    client_kwargs = {"base_url": base_url} if base_url else {}
    client = AsyncOpenAI(**client_kwargs)
    rows = _load_rows(input_path)
    task_map = _load_task_map(task_source)
    semaphore = asyncio.Semaphore(concurrency)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = _partial_output_path(output_path)

    if output_path.exists():
        logger.info("Found completed relabeled file at %s; skipping this phase", output_path)
        return 0

    existing_results: list[dict] = []
    if partial_path.exists():
        existing_results = _dedupe_rows(_load_rows(partial_path))
        with open(partial_path, "w") as f:
            for row in existing_results:
                f.write(json.dumps(row) + "\n")

    completed_keys = {_key(row) for row in existing_results}
    remaining_rows = [row for row in rows if _key(row) not in completed_keys]

    logger.info(
        "Relabeling %d raw rows from %s (%d already completed, %d remaining)",
        len(rows),
        input_path,
        len(existing_results),
        len(remaining_rows),
    )
    tasks = [
        asyncio.create_task(_relabel_one(client, model, row, task_map, default_source, semaphore))
        for row in remaining_rows
    ]
    results = list(existing_results)
    partial_written = len(existing_results)
    progress = (
        tqdm(total=len(tasks), desc=f"Relabel {default_source}", unit="row")
        if tqdm is not None
        else None
    )
    for idx, future in enumerate(asyncio.as_completed(tasks), start=1):
        try:
            result = await future
        except Exception as exc:
            logger.warning("Relabel task crashed for row %d: %s", idx, exc)
            result = None
        if isinstance(result, Exception):
            logger.warning("Relabel failed for row %d: %s", idx, result)
        elif result is not None:
            results.append(result)
            with open(partial_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            partial_written += 1
        if progress is not None:
            progress.update(1)
        elif idx % 10 == 0 or idx == len(remaining_rows):
            logger.info("Processed %d/%d", idx, len(remaining_rows))
        if idx % 50 == 0 or idx == len(remaining_rows):
            logger.info(
                "Relabel %s progress: %d/%d processed, %d usable rows written to %s",
                default_source,
                idx,
                len(remaining_rows),
                partial_written,
                partial_path,
            )
    if progress is not None:
        progress.close()

    for row in results:
        row.setdefault("split_group_id", row["instance_id"])

    if add_negatives:
        results = _add_negatives(results, ratio=negative_ratio, seed=seed)
    with open(output_path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    if partial_path.exists():
        partial_path.unlink()
    logger.info("Wrote %d relabeled rows to %s", len(results), output_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Relabel raw tool outputs into canonical rows")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--task-source", type=Path, default=None)
    parser.add_argument("--default-source", default="swe")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--add-negatives", action="store_true")
    parser.add_argument("--negative-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return asyncio.run(
        relabel_async(
            input_path=args.input,
            output_path=args.output,
            model=args.model,
            base_url=args.base_url,
            task_source=args.task_source,
            default_source=args.default_source,
            concurrency=args.concurrency,
            add_negatives=args.add_negatives,
            negative_ratio=args.negative_ratio,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
