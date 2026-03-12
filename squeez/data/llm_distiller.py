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

from squeez.data.config import (
    MAX_TOOL_OUTPUT_LINES,
    MAX_TOOL_PROMPT_LINE_CHARS,
    MAX_TOOL_PROMPT_TOTAL_CHARS,
    PipelineConfig,
)

logger = logging.getLogger(__name__)

DISTILLATION_PROMPT = """You are an expert coding assistant. An agent needs to prune one tool observation.

FOCUSED QUERY: {query}

BACKGROUND TASK:
{issue_text}

The agent ran this tool:
TOOL: {tool_type} {command}

Raw output ({total_lines} lines):
{numbered_output}

Which line ranges should the agent focus on? Return a JSON object with a "spans" array.
Each span has "start" (first line number), "end" (last line number, inclusive), and "reason" (brief explanation).

Rules:
- Select ONLY lines relevant to answering the focused query
- The kept text must directly answer the query using evidence that is actually visible in this tool output
- Respect the tool type:
  - read_file / git_diff: keep the smallest code block or diff hunk that answers the query
  - test_output / build_output / type_check / pip_install: keep the smallest failure or warning block that answers the query
  - grep / ls / git_log / git_blame: keep only the listing, hit, commit, or blame lines that answer the query
  - curl: keep the smallest returned docs/API block that answers the query
- Prefer 1-3 contiguous evidence blocks rather than scattered lines
- Include minimal local context when needed
- Prefer the smallest useful block, usually 1-20 lines per span
- Do not select broad context if a smaller block answers the query
- If the output seems completely unrelated to the task, return {{"spans": [], "summary": "brief explanation of why this is irrelevant"}}
- Return ONLY valid JSON, nothing else

Good examples:
- Query: "Find the failure block that explains the missing module during installation."
  Keep: the traceback or error block showing ModuleNotFoundError.
- Query: "Find the code block that normalizes FITS file modes."
  Keep: the function definition block, not the entire file.
- Query: "Find the commit entry most relevant to the CSRF referer change."
  Keep: one or a few git log lines, not the whole log.

Bad examples:
- Query asks for a class definition but the tool output is type_check stderr.
  Return no spans instead of inventing a code block.
- Query can be answered by one error line, but you keep 200 surrounding lines.
  This is too broad.
- Query asks for "all lines containing X".
  That is too lexical and should not happen; select only the evidence block that matters.

Example response:
{{"spans": [{{"start": 1, "end": 3, "reason": "relevant imports"}}, {{"start": 45, "end": 62, "reason": "buggy method"}}]}}"""


QUERY_PROMPT = """You are creating a context-pruning benchmark for a coding agent.

Original task:
{issue_text}

Tool:
{tool_type} {command}

Raw output:
{numbered_output}

Tool-specific guidance:
{tool_guidance}

Optional existing query hint:
{query_hint}

Write one short, focused extraction query for what evidence the agent should
keep from this one tool output.

Rules:
- Derive the query from the original task, but turn it into one realistic next-step subgoal for this specific tool output
- Ask for evidence the agent should read next, not a diagnosis or full bug fix
- The query must be answerable from this tool output alone
- Respect the tool type:
  - read_file / git_diff: ask for the smallest code block or diff hunk relevant to the task
  - test_output / build_output / type_check / pip_install: ask for the smallest failure/warning block that explains the issue
  - grep / ls / git_log / git_blame: ask for the most relevant hit, file, commit entry, or blame block
  - curl: ask for the returned docs/API block relevant to the task
- Avoid trivial lexical queries like:
  - "find all lines containing X"
  - "find the function named foo"
  - "find the import of bar"
- Avoid impossible query/tool mismatches:
  - do not ask a type_check log for a class definition
  - do not ask ls output for an error traceback or to read code inside a file
- Prefer semantic phrasing tied to the task, for example:
  - "Find the failure block that explains the missing dependency during install"
  - "Find the code block responsible for normalizing FITS file modes"
  - "Find the commit entry most relevant to the world_to_pixel_values behavior"
  - "Find the grep hits most relevant to uncertainty conversion"
- Keep it to one sentence
- The query should be easy enough for a human reviewer to judge, but not so literal that plain grep would make the task meaningless

Good examples:
- Tool: read_file
  Original task: fix a bug in FITS mode normalization
  Good query: "Find the code block that normalizes FITS file modes."
- Tool: pip_install
  Original task: fix install failure in astropy build
  Good query: "Find the failure block that explains the missing setuptools module during installation."
- Tool: git_log
  Original task: fix a regression in world_to_pixel_values
  Good query: "Find the commit entry most relevant to world_to_pixel_values behavior."
- Tool: grep
  Original task: fix uncertainty conversion behavior
  Good query: "Find the grep hits most relevant to uncertainty conversion."
- Tool: ls
  Original task: fix uncertainty conversion behavior
  Good query: "Find the file entries most relevant to uncertainty handling in astropy.nddata."

Bad examples:
- "Find all lines containing 'ModuleNotFoundError'"
- "Find the _normalize_fits_mode function definition block"
- "Find the import statement for nduncertainty"
- "Find the BlackBody class definition" when the tool output is type_check stderr
- "Read the code in astropy/nddata/nduncertainty.py" when the tool output is ls

Return JSON only:
{{"query": "Find the ... block ..."}}
"""


CONTENT_FIRST_QUERY_PROMPT = """You are creating a context-pruning benchmark for a coding agent.

The original task is only soft background:
{issue_text}

Tool:
{tool_type} {command}

Raw output:
{numbered_output}

Write one short extraction query that is clearly answerable from this tool output alone.

Rules:
- Prioritize what is actually visible in this tool output over the original task description
- If the original task seems mismatched to this output, ignore it and write the best query for the useful evidence in this output
- Respect the tool type:
  - read_file / git_diff: ask for the smallest code block or diff hunk that seems most useful
  - test_output / build_output / type_check / pip_install / python: ask for the smallest failure or runtime block
  - grep / ls / git_log / git_blame: ask for the most useful hit, file entry, commit entry, or blame block
  - curl: ask for the most useful returned docs/API block
- Avoid trivial lexical phrasing like "find all lines containing X"
- Avoid impossible asks, such as asking ls output to read code inside a file
- Keep it to one sentence

Good examples:
- read_file: "Find the code block that normalizes FITS file modes."
- python: "Find the runtime error block that explains the missing setuptools_scm module."
- ls: "Find the file entry most relevant to uncertainty handling in astropy.nddata."
- git_log: "Find the commit entry most relevant to allowing wcs=None in CCDData."

Return JSON only:
{{"query": "Find the ... block ..."}}
"""

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry

_TOOL_GUIDANCE = {
    "read_file": (
        "Ask for the smallest code block that explains the behavior, bug, or API handling "
        "mentioned in the task. Function or method names are acceptable only when they are "
        "used to describe behavior, not as pure lookup targets."
    ),
    "git_diff": (
        "Ask for the smallest diff hunk relevant to the bug or behavior change. "
        "Do not ask for generic code definitions."
    ),
    "grep": (
        "Ask for the most relevant grep hits for the task. Do not ask for 'all lines containing X'. "
        "The answer should be a small subset of hits, not a broad lexical dump."
    ),
    "ls": (
        "Ask for the most relevant file or directory entries for the task. "
        "Do not ask for stack traces, code blocks, or full diagnosis."
    ),
    "git_log": (
        "Ask for the most relevant commit entry or small set of entries for the task. "
        "Do not ask for code definitions or broad history."
    ),
    "git_blame": (
        "Ask for the most relevant blame block or lines for the task. "
        "Keep the query about authorship/history around a behavior, not full code definitions."
    ),
    "test_output": (
        "Ask for the failure block or error block that best explains the task. "
        "Do not ask for code definitions."
    ),
    "build_output": (
        "Ask for the build failure block or warning block most relevant to the task. "
        "Do not ask for code definitions."
    ),
    "type_check": (
        "Ask for the type-check failure block, error message, or diagnostic block most relevant "
        "to the task. Never ask this tool output for a class, function, or import definition."
    ),
    "lint_output": ("Ask for the lint failure block or warning block most relevant to the task."),
    "pip_install": (
        "Ask for the dependency or installation failure block most relevant to the task. "
        "Never ask for code definitions."
    ),
    "python": (
        "Ask for the runtime error block, traceback, or printed evidence block most relevant to the task. "
        "Never ask for code definitions."
    ),
    "curl": (
        "Ask for the returned docs or API block most relevant to the task. "
        "Do not ask for code not present in the response."
    ),
    "coverage": ("Ask for the coverage lines or summary block most relevant to the task."),
}

_TOOL_FORBIDDEN_MARKERS = {
    "type_check": (
        "class definition",
        "function definition",
        "method definition",
        "import statement",
    ),
    "python": ("class definition", "function definition", "method definition", "import statement"),
    "pip_install": (
        "class definition",
        "function definition",
        "method definition",
        "import statement",
    ),
    "build_output": (
        "class definition",
        "function definition",
        "method definition",
        "import statement",
    ),
    "test_output": (
        "class definition",
        "function definition",
        "method definition",
        "import statement",
    ),
    "ls": (
        "traceback",
        "error block",
        "class definition",
        "function definition",
        "read the code",
        "code block",
        "implementation of",
    ),
    "git_log": ("class definition", "function definition", "import statement"),
}


def _tool_guidance(tool_type: str) -> str:
    return _TOOL_GUIDANCE.get(
        tool_type,
        "Ask for the smallest evidence block this tool output can actually reveal.",
    )


def _fallback_query(tool_type: str, query_hint: str | None = None) -> str:
    if query_hint:
        return query_hint
    if tool_type == "read_file":
        return "Find the smallest code block most relevant to the task."
    if tool_type == "git_diff":
        return "Find the smallest diff hunk most relevant to the task."
    if tool_type == "grep":
        return "Find the grep hits most relevant to the task."
    if tool_type == "ls":
        return "Find the file or directory entries most relevant to the task."
    if tool_type == "git_log":
        return "Find the commit entry most relevant to the task."
    if tool_type == "git_blame":
        return "Find the blame block most relevant to the task."
    if tool_type in {"test_output", "build_output", "lint_output", "type_check", "pip_install"}:
        return "Find the failure block most relevant to the task."
    if tool_type == "python":
        return "Find the runtime output block most relevant to the task."
    if tool_type == "curl":
        return "Find the returned docs or API block most relevant to the task."
    return "Find the evidence block most relevant to the task."


def _query_is_compatible(tool_type: str, query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    if "all lines containing" in q:
        return False
    if q.startswith("find the import statement") and tool_type != "read_file":
        return False
    if tool_type in _TOOL_FORBIDDEN_MARKERS:
        if any(marker in q for marker in _TOOL_FORBIDDEN_MARKERS[tool_type]):
            return False
    if tool_type in {"type_check", "python", "build_output", "pip_install", "test_output"}:
        if "code block" in q and any(
            term in q for term in ("class", "function", "method", "import")
        ):
            return False
    return True


def _number_output_for_prompt(
    text: str,
    *,
    max_lines: int = MAX_TOOL_OUTPUT_LINES,
    max_chars_per_line: int = MAX_TOOL_PROMPT_LINE_CHARS,
    max_total_chars: int = MAX_TOOL_PROMPT_TOTAL_CHARS,
) -> str:
    """Build a numbered teacher view with bounded line/character budgets."""
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


def _parse_query_response(raw_response: str) -> str | None:
    """Parse the teacher's focused-query response."""
    text = raw_response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None
    query = data.get("query")
    if not isinstance(query, str):
        return None
    query = query.strip()
    return query or None


async def _generate_query(
    client: AsyncOpenAI,
    sample: dict,
    issue_text: str,
    model: str,
) -> str | None:
    """Generate a focused extraction query from the original task and raw output."""
    existing_query = sample.get("query")
    prompt = QUERY_PROMPT.format(
        issue_text=issue_text,
        tool_type=sample["tool_type"],
        command=sample.get("command", ""),
        numbered_output=_number_output_for_prompt(sample["output"]),
        tool_guidance=_tool_guidance(sample["tool_type"]),
        query_hint=existing_query or "(none)",
    )
    for attempt in range(MAX_RETRIES):
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
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BASE_DELAY * (2**attempt))
                continue
            return None
        raw = raw_content.strip()
        parsed = _parse_query_response(raw)
        if parsed and _query_is_compatible(sample["tool_type"], parsed):
            return parsed
        if parsed:
            logger.debug(
                "Rejected incompatible query for %s/%s: %r",
                sample["instance_id"],
                sample["tool_type"],
                parsed,
            )
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_BASE_DELAY * (2**attempt))
    return _fallback_query(sample["tool_type"], existing_query)


async def _generate_content_first_query(
    client: AsyncOpenAI,
    sample: dict,
    issue_text: str,
    model: str,
) -> str:
    """Generate a query driven primarily by the tool output itself."""
    existing_query = sample.get("query")
    prompt = CONTENT_FIRST_QUERY_PROMPT.format(
        issue_text=issue_text,
        tool_type=sample["tool_type"],
        command=sample.get("command", ""),
        numbered_output=_number_output_for_prompt(sample["output"]),
    )
    for attempt in range(MAX_RETRIES):
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
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BASE_DELAY * (2**attempt))
                continue
            break
        raw = raw_content.strip()
        parsed = _parse_query_response(raw)
        if parsed and _query_is_compatible(sample["tool_type"], parsed):
            return parsed
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_BASE_DELAY * (2**attempt))
    return _fallback_query(sample["tool_type"], existing_query)


async def _distill_single(
    client: AsyncOpenAI,
    sample: dict,
    instance: dict,
    model: str,
    temperature: float,
    generate_query: bool,
    semaphore: asyncio.Semaphore,
    counter: dict,
) -> dict | None:
    """Distill a single sample with retry logic."""
    async with semaphore:
        issue_text = instance.get("problem_statement", "")
        if len(issue_text) > 3000:
            issue_text = issue_text[:3000] + "..."
        query = sample.get("query")
        if generate_query:
            try:
                generated_query = await _generate_query(client, sample, issue_text, model)
                if generated_query:
                    query = generated_query
            except Exception as e:
                logger.debug(f"Focused query generation failed for {sample['instance_id']}: {e}")
        query = query or issue_text[:200]

        prompt = DISTILLATION_PROMPT.format(
            issue_text=issue_text,
            query=query,
            tool_type=sample["tool_type"],
            command=sample.get("command", ""),
            total_lines=sample.get("num_total_lines", sample.get("num_lines", 0)),
            numbered_output=_number_output_for_prompt(sample["output"]),
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You select relevant line ranges from tool output. Always respond with valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=8192,
                )

                raw_content = response.choices[0].message.content
                if raw_content is None:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_BASE_DELAY * (2**attempt))
                        continue
                    counter["failed"] += 1
                    return None

                raw = raw_content.strip()
                spans, summary = _parse_spans_response(raw)

                original_lines = sample.get("num_total_lines", sample.get("num_lines", 1))

                # For the main positive benchmark, empties should usually be
                # dropped rather than preserved as labels. Retry once with a
                # content-first query before giving up.
                if not spans and summary:
                    if sample.get("source") != "synthetic_negative":
                        content_query = await _generate_content_first_query(
                            client,
                            sample,
                            issue_text,
                            model,
                        )
                        if content_query and content_query != query:
                            query = content_query
                            prompt = DISTILLATION_PROMPT.format(
                                issue_text=issue_text,
                                query=query,
                                tool_type=sample["tool_type"],
                                command=sample.get("command", ""),
                                total_lines=sample.get(
                                    "num_total_lines", sample.get("num_lines", 0)
                                ),
                                numbered_output=_number_output_for_prompt(sample["output"]),
                            )
                            if attempt < MAX_RETRIES - 1:
                                await asyncio.sleep(RETRY_BASE_DELAY * (2**attempt))
                                continue
                    counter["failed"] += 1
                    return None

                if not spans:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_BASE_DELAY * (2**attempt))
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

                kept_lines = sum(
                    s.get("end", s.get("start", 0)) - s.get("start", 0) + 1 for s in spans
                )
                counter["done"] += 1
                return {
                    **sample,
                    "distilled_output": distilled,
                    "distilled_lines": distilled_lines,
                    "compression_ratio": round(compression, 4),
                    "spans": spans,
                    "gold_spans": [
                        {"start_line": span["start"], "end_line": span["end"]} for span in spans
                    ],
                    "kept_lines": kept_lines,
                    "background_task": issue_text,
                    "query": query,
                    "tool_output": sample["output"],
                    "is_irrelevant": False,
                }

            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "429" in err_str or "too many" in err_str:
                    delay = RETRY_BASE_DELAY * (2**attempt) * 2
                    logger.warning(f"Rate limited, retrying in {delay:.0f}s...")
                    await asyncio.sleep(delay)
                    continue
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2**attempt))
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

    logger.info(
        f"Distilling {len(samples)} samples with {config.distillation_model} "
        f"(concurrency={config.distillation_max_concurrent})..."
    )
    start_time = time.time()

    for batch_start in range(0, len(samples), BATCH_SIZE):
        batch = samples[batch_start : batch_start + BATCH_SIZE]
        counter["total"] = batch_start + len(batch)

        tasks = []
        for sample in batch:
            instance = instance_map.get(sample["instance_id"])
            if not instance:
                continue
            # Store task text so downstream consumers don't need instance lookup
            issue_text = instance.get("problem_statement", "")
            if len(issue_text) > 3000:
                issue_text = issue_text[:3000] + "..."
            sample["task"] = issue_text
            sample["background_task"] = issue_text
            sample["query"] = sample.get("query") or issue_text[:200]
            tasks.append(
                _distill_single(
                    client,
                    sample,
                    instance,
                    config.distillation_model,
                    config.distillation_temperature,
                    semaphore,
                    counter,
                    config.generate_queries_with_teacher,
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

    logger.info(
        f"Successfully distilled {len(all_results)}/{len(samples)} samples "
        f"in {time.time() - start_time:.1f}s"
    )
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
        s for s in labeled_samples if f"{s['instance_id']}::{s.get('command', '')}" not in done_keys
    ]

    if not remaining:
        logger.info(f"All {len(existing)} samples already distilled")
        return existing

    logger.info(f"Resuming distillation: {len(existing)} done, {len(remaining)} remaining")

    new_results = asyncio.run(distill_batch_async(remaining, instances, config, str(output_path)))

    all_results = existing + new_results

    # Final save (clean rewrite)
    with open(output_path, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")

    logger.info(f"Saved {len(all_results)} distilled samples to {output_path}")
    return all_results
