"""Phase 6: LLM distillation — teacher model selects relevant line spans.

For each tool output, sends the raw output + task description to the teacher
model. The teacher returns JSON with line ranges to keep. We then match those
ranges against the original output to build the filtered result — guaranteeing
zero hallucination in training targets.

Handles high-throughput batching with:
- Configurable concurrency (default 50)
- Exponential backoff retry on rate limits
- Incremental saving (no lost work on crash)
- Progress logging
"""

import asyncio
import json
import logging
import os
import re
import time

from openai import AsyncOpenAI

from squeez.data.config import PipelineConfig

logger = logging.getLogger(__name__)

DISTILLATION_PROMPT = """You are an expert coding assistant. An agent is working on this issue:

TASK: {issue_text}

The agent ran this tool:
TOOL: {tool_type} {command}

Raw output ({total_lines} lines):
{numbered_output}

Which line ranges should the agent focus on? Return a JSON object with a "spans" array.
Each span has "start" (first line number), "end" (last line number, inclusive), and "reason" (brief explanation).

Rules:
- Select ONLY lines relevant to understanding or solving the issue
- Include relevant imports, class/function signatures, error messages, and suspicious code
- Target: keep 5-30% of lines
- If the output seems completely unrelated to the task, return {{"spans": [], "summary": "brief explanation of why this is irrelevant"}}
- Return ONLY valid JSON, nothing else

Example response:
{{"spans": [{{"start": 1, "end": 3, "reason": "relevant imports"}}, {{"start": 45, "end": 62, "reason": "buggy method"}}]}}"""

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry


def _build_filtered_output(original_output: str, spans: list[dict]) -> str:
    """Build filtered output by extracting span ranges from original.

    This guarantees zero hallucination — every line in the output
    comes directly from the original tool output.
    """
    lines = original_output.split("\n")
    total = len(lines)

    if not spans:
        return ""

    # Sort spans by start line and clamp to valid range
    valid_spans = []
    for span in spans:
        start = max(1, span.get("start", 1))
        end = min(total, span.get("end", start))
        if start <= end:
            valid_spans.append((start, end))

    if not valid_spans:
        return ""

    # Merge overlapping spans
    valid_spans.sort()
    merged = [valid_spans[0]]
    for start, end in valid_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    # Build output with omission markers
    result_lines = []
    prev_end = 0
    for start, end in merged:
        omitted = start - prev_end - 1
        if omitted > 0 and prev_end > 0:
            result_lines.append(f"... ({omitted} lines omitted)")
        elif omitted > 0 and prev_end == 0 and start > 1:
            result_lines.append(f"... ({start - 1} lines omitted)")

        for i in range(start - 1, end):  # 0-indexed
            if i < total:
                result_lines.append(lines[i])
        prev_end = end

    # Trailing omission
    if prev_end < total:
        omitted = total - prev_end
        result_lines.append(f"... ({omitted} lines omitted)")

    return "\n".join(result_lines)


def _parse_spans_response(raw_response: str) -> tuple[list[dict], str | None]:
    """Parse the LLM's JSON response, handling common formatting issues."""
    text = raw_response.strip()

    # Strip code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return [], None
        else:
            return [], None

    spans = data.get("spans", [])
    summary = data.get("summary")
    return spans, summary


async def _distill_single(
    client: AsyncOpenAI,
    sample: dict,
    instance: dict,
    model: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    counter: dict,
) -> dict | None:
    """Distill a single sample with retry logic."""
    async with semaphore:
        issue_text = instance.get("problem_statement", "")
        if len(issue_text) > 3000:
            issue_text = issue_text[:3000] + "..."

        prompt = DISTILLATION_PROMPT.format(
            issue_text=issue_text,
            tool_type=sample["tool_type"],
            command=sample.get("command", ""),
            total_lines=sample.get("num_total_lines", sample.get("num_lines", 0)),
            numbered_output=sample["output"],
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You select relevant line ranges from tool output. Always respond with valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=2048,
                )

                raw = response.choices[0].message.content.strip()
                spans, summary = _parse_spans_response(raw)

                original_lines = sample.get("num_total_lines", sample.get("num_lines", 1))

                # If no spans and a summary — the output is irrelevant
                if not spans and summary:
                    distilled = f"[Not relevant to task: {summary}]"
                    counter["done"] += 1
                    return {
                        **sample,
                        "distilled_output": distilled,
                        "distilled_lines": 1,
                        "compression_ratio": round(1.0 - (1 / original_lines), 4),
                        "spans": [],
                        "summary": summary,
                    }

                if not spans:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                        continue
                    counter["failed"] += 1
                    return None

                # Build filtered output from original lines
                distilled = _build_filtered_output(sample["output"], spans)

                if not distilled.strip():
                    counter["failed"] += 1
                    return None

                distilled_lines = len(distilled.split("\n"))
                compression = 1.0 - (distilled_lines / original_lines)

                if compression < 0:
                    counter["failed"] += 1
                    return None

                kept_lines = sum(s.get("end", s.get("start", 0)) - s.get("start", 0) + 1 for s in spans)
                counter["done"] += 1
                return {
                    **sample,
                    "distilled_output": distilled,
                    "distilled_lines": distilled_lines,
                    "compression_ratio": round(compression, 4),
                    "spans": spans,
                    "kept_lines": kept_lines,
                }

            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "429" in err_str or "too many" in err_str:
                    delay = RETRY_BASE_DELAY * (2 ** attempt) * 2
                    logger.warning(f"Rate limited, retrying in {delay:.0f}s...")
                    await asyncio.sleep(delay)
                    continue
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                logger.error(f"Distillation error for {sample['instance_id']}: {e}")
                counter["failed"] += 1
                return None

        counter["failed"] += 1
        return None


async def distill_batch_async(
    samples: list[dict],
    instances: list[dict],
    config: PipelineConfig,
    output_path: str | None = None,
) -> list[dict]:
    """Distill all samples with high-throughput async batching.

    Features:
    - Concurrent requests controlled by config.distillation_max_concurrent
    - Retry with exponential backoff on rate limits
    - Incremental saving every 100 results
    - Progress logging every 50 completions
    """
    api_key = config.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("API key not set (use --api-key or OPENAI_API_KEY)")

    client_kwargs = {"api_key": api_key}
    if config.distillation_base_url:
        client_kwargs["base_url"] = config.distillation_base_url
    client = AsyncOpenAI(**client_kwargs)
    semaphore = asyncio.Semaphore(config.distillation_max_concurrent)
    instance_map = {inst["instance_id"]: inst for inst in instances}

    counter = {"done": 0, "failed": 0, "total": 0}
    all_results: list[dict] = []

    # Process in batches for incremental saving
    BATCH_SIZE = 200

    logger.info(f"Distilling {len(samples)} samples with {config.distillation_model} "
                f"(concurrency={config.distillation_max_concurrent})...")
    start_time = time.time()

    for batch_start in range(0, len(samples), BATCH_SIZE):
        batch = samples[batch_start:batch_start + BATCH_SIZE]
        counter["total"] = batch_start + len(batch)

        tasks = []
        for sample in batch:
            instance = instance_map.get(sample["instance_id"])
            if not instance:
                continue
            tasks.append(
                _distill_single(
                    client, sample, instance,
                    config.distillation_model,
                    config.distillation_temperature,
                    semaphore, counter,
                )
            )

        results = await asyncio.gather(*tasks)
        batch_results = [r for r in results if r is not None]
        all_results.extend(batch_results)

        elapsed = time.time() - start_time
        rate = counter["done"] / elapsed if elapsed > 0 else 0
        eta = (len(samples) - counter["done"]) / rate if rate > 0 else 0
        logger.info(
            f"Progress: {counter['done']}/{len(samples)} done, "
            f"{counter['failed']} failed, "
            f"{rate:.1f}/s, ETA {eta:.0f}s"
        )

        # Incremental save
        if output_path and all_results:
            with open(output_path, "w") as f:
                for result in all_results:
                    f.write(json.dumps(result) + "\n")

    logger.info(f"Successfully distilled {len(all_results)}/{len(samples)} samples "
                f"in {time.time() - start_time:.1f}s")
    return all_results


def distill_all(
    labeled_samples: list[dict],
    instances: list[dict],
    config: PipelineConfig,
) -> list[dict]:
    """Distill all tool outputs (sync wrapper).

    Supports resuming from partial results — skips samples already distilled.
    """
    output_path = config.output_dir / "distilled_outputs.jsonl"

    # Load existing partial results for resume
    existing = []
    done_keys: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    existing.append(r)
                    # Unique key: instance_id + command
                    done_keys.add(f"{r['instance_id']}::{r.get('command', '')}")

    # Filter to only undone samples
    remaining = [
        s for s in labeled_samples
        if f"{s['instance_id']}::{s.get('command', '')}" not in done_keys
    ]

    if not remaining:
        logger.info(f"All {len(existing)} samples already distilled")
        return existing

    logger.info(f"Resuming distillation: {len(existing)} done, {len(remaining)} remaining")

    new_results = asyncio.run(
        distill_batch_async(remaining, instances, config, str(output_path))
    )

    all_results = existing + new_results

    # Final save (clean rewrite)
    with open(output_path, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")

    logger.info(f"Saved {len(all_results)} distilled samples to {output_path}")
    return all_results
