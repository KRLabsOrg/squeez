---
title: "Squeez: Task-Conditioned Tool-Output Pruning for Coding Agents"
emoji: "🗜️"
colorFrom: "indigo"
colorTo: "blue"
sdk: "static"
pinned: false
license: "apache-2.0"
---

# How We Built a Tool-Output Pruner for Coding Agents

<p align="center">
  <img src="./assets/squeez_mascot.png" alt="Squeez mascot" width="180">
</p>

We trained and open-sourced **Squeez-2B**, a compact model for pruning tool output in coding agents. Given a focused query and one raw tool observation, it returns the smallest verbatim evidence block that the agent should inspect next. On our held-out benchmark it reaches **0.862 recall at 91.5% compression**, outperforming a zero-shot **Qwen 3.5 35B A3B** baseline by **11.3 recall points** while operating at essentially the same compression level.

The release consists of three parts:

- Model: [KRLabsOrg/squeez-2b](https://huggingface.co/KRLabsOrg/squeez-2b)
- Dataset: [KRLabsOrg/tool-output-extraction-swebench](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)
- Code and CLI: [github.com/KRLabsOrg/squeez](https://github.com/KRLabsOrg/squeez)

This post describes the problem, explains how we built a benchmark for it, and shows that dedicated supervision works substantially better than larger zero-shot models or simple retrieval heuristics.

## The Problem

Coding agents such as Claude Code and Codex spend much of their time reading tool output. When an agent runs `pytest`, `grep`, `git log`, `kubectl`, or `pip install`, the result is often dozens or hundreds of lines long. Only a small fraction of those lines matter for the next step. The rest is headers, passing tests, repeated metadata, timestamps, unchanged context, or structurally similar but irrelevant matches. In practice, this means that a substantial part of the context budget is consumed not by reasoning, but by re-reading noisy observations.

This is easiest to see on test output. An agent runs `pytest`, receives a moderately long result, and only one failure block matters:

**Raw tool output (45 lines):**

```text
$ python -m pytest tests/ -v
======================== test session starts ===========
platform linux -- Python 3.12.1, pytest-8.1.1
collected 23 items

tests/test_auth.py::test_login_valid PASSED
tests/test_auth.py::test_login_invalid PASSED
tests/test_auth.py::test_token_refresh FAILED
tests/test_auth.py::test_logout PASSED
tests/test_users.py::test_create_user PASSED
  ... 6 more PASSED ...
tests/test_middleware.py::test_cors_headers PASSED

======================= FAILURES =======================
_____ test_token_refresh _______________________________

    def test_token_refresh(self):
        token = self.client.get_token(expired=True)
>       refreshed = self.client.refresh(token)
E       AuthenticationError: Token refresh window expired
E       Expected: new token within 30m window
E       Got: rejection after 15m (timeout changed?)

tests/test_auth.py:47: AuthenticationError
================ short test summary info ===============
FAILED tests/test_auth.py::test_token_refresh
================== 1 failed, 9 passed =================
```

**After Squeez (6 lines):**

```text
tests/test_auth.py::test_token_refresh FAILED

    def test_token_refresh(self):
        token = self.client.get_token(expired=True)
>       refreshed = self.client.refresh(token)
E       AuthenticationError: Token refresh window expired
E       Expected: new token within 30m window
E       Got: rejection after 15m (timeout changed?)
```

That is **87% compression** while preserving the only part of the observation that matters for the next debugging step:

```bash
python -m pytest tests/ -v 2>&1 | squeez "find the test failure related to authentication"
```

The same pattern appears in many other tools. `grep` may return a long list of nearby lexical matches although only one file is relevant. `git log` may show a long history where one commit matters. `kubectl describe` may contain hundreds of lines of pod state, yet the evidence is two lines saying `OOMKilled` and `Exit Code: 137`. `read_file` may return an entire module even though the agent only needs one code block. The common structure is always the same: a small evidence block embedded in a much larger observation.

Existing pruning systems point in the right direction, but usually operate on different units. **LLMLingua** and **LongLLMLingua** compress prompts at the token or prompt-block level ([Jiang et al., 2023](https://aclanthology.org/2023.emnlp-main.825/); [Jiang et al., 2024](https://aclanthology.org/2024.acl-long.91/)). **EXIT** and **Provence** perform extractive compression over retrieved text for downstream question answering or retrieval-augmented generation ([Hwang et al., 2025](https://aclanthology.org/2025.findings-acl.253/); [Chirkova et al., 2025](https://arxiv.org/abs/2501.16214)). **Zilliz Semantic Highlight** adapts this line to semantic highlighting over retrieved passages ([model card](https://huggingface.co/zilliz/semantic-highlight-bilingual-v1)). **SWE-Pruner** is the closest coding baseline, but it focuses on pruning repository code context rather than a single mixed-format tool observation ([Wang et al., 2026](https://arxiv.org/abs/2601.16746)).

Tool output is a different object. It is not well-formed prose, and it is not always source code. A single observation may mix code, logs, shell traces, stack frames, JSON payloads, and Git metadata. The relevant unit may be a failure block, a short function body, a commit entry, a package conflict, or nothing at all. That is the gap Squeez targets.

## The Task and the Benchmark

We formulate the problem as **task-conditioned tool-output pruning**: given a focused query and one raw tool observation, return the smallest verbatim evidence block that the agent should inspect next. The model is not asked to solve the full bug from one observation. It is asked to preserve the relevant evidence and remove the rest.

Two properties of the task matter. First, the output is **verbatim**. We do not want paraphrased summaries of stack traces, imports, versions, exit codes, or code blocks. Tool output often contains details that should remain exact. Second, the query is **task-conditioned** but narrower than the full issue description. It expresses the local information need the agent has at that moment: find the failure block, the relevant code region, or the commit that likely introduced the behavior.

The overall pipeline is shown below:

<p align="center">
  <img src="./assets/squeez_overview.svg" alt="Squeez pipeline: from raw tool output through span annotation to generative model" width="920">
</p>

The benchmark is built from two sources. The first is [SWE-bench](https://openreview.net/forum?id=VTF8yNQM66), which provides real GitHub issue-resolution tasks over real repositories. We do not use SWE-bench as another patch-generation benchmark. Instead, we use it as a source of realistic repository snapshots, issue contexts, and raw tool observations. Starting from cloned SWE-bench repositories, we collected or reused **10,713** raw tool observations, including file reads, grep hits, Git history, shell output, test results, Python exceptions, and package-manager traces.

The second source is synthetic multi-ecosystem tool output. Its role is to broaden coverage where SWE-bench is thin, especially outside the Python-heavy distribution of repository-level issue fixing. Starting from **2,039** raw synthetic observations, we add examples from TypeScript, Go, Rust, Java, Docker, Terraform, and Kubernetes workflows, and we also construct explicit negatives where the correct pruning decision is to return nothing.

Each released example is built with a two-stage teacher-labeling pipeline using `openai/gpt-oss-120b`. First, the teacher writes a focused extraction query for one observation. Second, it selects the smallest contiguous span or set of spans that answers that query. The teacher sees a numbered rendering of the output for stable span selection, but the released labels are always mapped back onto the original raw text. This is a deliberate design choice: the benchmark stores a pruning decision over the source observation, not a free-form textual explanation of it. Positive examples whose query cannot be supported by the observation are dropped rather than retained as accidental empty outputs. Explicit negatives are created separately in the synthetic portion.

The held-out set was manually curated. Starting from **729** candidate test examples, we removed **111** cases (15.2%) that were near-duplicates, trivial 1--2 line outputs, overly broad spans, or incorrect annotations. The final test set contains **618** manually reviewed examples.

The released benchmark contains **11,477** examples in total: **9,205** SWE-derived examples, **1,697** synthetic positives, and **575** synthetic negatives. SWE-derived examples are split by repository and synthetic examples by tool family.

| Source | Raw inputs | Released rows |
|---|---:|---:|
| SWE-derived | 10,713 | 9,205 |
| Synthetic positives | 2,039 | 1,697 |
| Synthetic negatives | — | 575 |
| Total | 12,752 | 11,477 |

The benchmark covers **27** tool types. The largest families are shown below.

| Tool family | Rows | Avg. input | Avg. gold |
|---|---:|---:|---:|
| `read_file` | 3768 | 1677 | 84 |
| `grep` | 1330 | 779 | 19 |
| `git_log` | 720 | 161 | 11 |
| `python` | 698 | 60 | 28 |
| `test_output` | 546 | 56 | 23 |
| `curl` | 493 | 723 | 68 |
| `pip_install` | 441 | 438 | 79 |
| `type_check` | 317 | 3418 | 39 |
| `git_blame` | 291 | 4210 | 139 |
| remaining tools | 3873 | 688 | 47 |

The distribution is intentionally heterogeneous. `python` and `test_output` rows are short; `read_file`, `type_check`, and `git_blame` can be extremely long. This variation is one reason simple truncation and lexical retrieval perform poorly: the useful evidence does not follow one structural pattern, and it may occur at the beginning, middle, or end of the observation.

## Training a Small Model for a Narrow Task

We chose **Qwen 3.5 2B** as the base model ([Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)). The choice was deliberate. The goal here is not to maximize zero-shot reasoning with the largest possible decoder. It is to learn a narrow supervised extraction policy that can run cheaply inside an agent loop. A dense 2B model is large enough to benefit from supervision, but still small enough to be practical for local serving and repeated tool use.

We fine-tuned the model with **LoRA** ([Hu et al., 2022](https://openreview.net/forum?id=nZeVKeeFYf9); [Dettmers et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)) using the **Unsloth** stack. The model receives a focused extraction query and the raw tool observation, and is trained to emit the extracted evidence wrapped in `<relevant_lines>` tags. In other words, the supervision target is not a classification label and not a summary. It is the exact evidence block the model should keep.

Training uses max sequence length 20,000, effective batch size 32, learning rate 2e-4, 3 epochs, warmup 0.05, weight decay 0.01. After training, we merge the LoRA adapter into the base model and serve the merged checkpoint through **vLLM**.

## Results

We compare Squeez-2B against three zero-shot generative baselines and four heuristic baselines. The heuristic baselines keep roughly 10% of the input lines to operate at a compression level similar to the gold extractions. The main metrics are **recall**, **F1**, and **compression**. Recall matters most because dropping relevant evidence is usually more harmful than keeping a slightly larger block.

| Model | Recall | F1 | Compression |
|---|---:|---:|---:|
| **Squeez-2B** | **0.8624** | **0.8035** | 0.9150 |
| Qwen 3.5 35B A3B | 0.7498 | 0.7254 | 0.9177 |
| Kimi K2 | 0.5286 | 0.6827 | 0.9425 |
| Qwen 3.5 2B (base) | 0.5299 | 0.5482 | 0.8197 |
| BM25 (10%) | 0.2172 | 0.2314 | 0.9036 |
| First-N (10%) | 0.1445 | 0.1570 | 0.9055 |
| Random (10%) | 0.1009 | 0.1966 | 0.9067 |
| Last-N (10%) | 0.0503 | 0.1393 | 0.9130 |

Three results matter most. First, **task-specific training matters**: a fine-tuned 2B model outperforms the 18x larger Qwen 3.5 35B A3B by **11.3 recall points** at almost the same compression level. Second, **heuristics are not sufficient**: BM25 reaches only **0.22 recall**, because lexical overlap is a poor proxy for relevance in stack traces, logs, and mixed-format observations. Third, **aggressive compression alone is not enough**: Kimi K2 removes the largest fraction of tokens, but pays for that compression with a large recall drop.

The recall-compression trade-off is shown below. Squeez-2B occupies the upper-left region: high recall with strong compression.

<p align="center">
  <img src="./assets/squeez_results_chart.svg" alt="Recall vs compression across all models" width="920">
</p>

The aggregate numbers are only part of the story. Qualitatively, the model appears to learn tool-specific pruning regularities. In `grep` and `git_log`, it tends to return the single relevant hit rather than a broader lexical neighborhood. In `test_output`, `build_output`, and package-manager logs, it keeps the failure block and drops surrounding boilerplate. In `read_file`, it often retains the smallest contiguous code block that answers the query instead of an entire surrounding function or class.

The following `kubectl` example illustrates the intended use case. The full observation contains 250 lines of pod description; the relevant evidence is a two-line block reporting `OOMKilled` and the exit code.

<p align="center">
  <img src="./assets/squeez_qualitative_example.svg" alt="kubectl example: 2 relevant lines from 250" width="920">
</p>

The strongest remaining failures are usually semantically adjacent but incorrect selections: choosing the wrong file from an `ls` listing, or returning a related commit that touches the same module without directly answering the query.

## Using Squeez

Operationally, Squeez is meant to be a preprocessing step rather than a new agent architecture. It does not require changes to the planner, tool API, or interaction loop. You can pipe tool output through the CLI:

```bash
pytest -q 2>&1 | squeez "find the failure block"
git log --oneline -50 | squeez "find the commit that changed CSRF handling"
cat src/auth/middleware.py | squeez "find the referer validation logic"
```

Or you can serve the model with vLLM for higher-throughput settings:

```bash
vllm serve KRLabsOrg/squeez-2b --dtype bfloat16 --max-model-len 16384
export SQUEEZ_SERVER_URL=http://localhost:8000/v1
pytest -q 2>&1 | squeez "find the failure block"
```

For systems such as Claude Code, a minimal `CLAUDE.md` instruction is enough:

```text
When you invoke a shell command, pipe it through `squeez` and describe what you need.
Examples:
- `bun test 2>&1 | squeez "did the tests pass?"`
- `git log --oneline -50 | squeez "find the commit that broke CSRF"`
```

The same pattern works with Codex and other agent setups that accept system-level instructions or shell wrappers.

## Takeaway

One recurring bottleneck in coding agents is deciding what to keep from a single tool observation. Our results suggest this is learnable, practically useful, and not handled well by simple heuristics or larger zero-shot models alone. Squeez is our attempt at a focused solution: a narrow model for a narrow problem.

## Resources

- **Model:** [KRLabsOrg/squeez-2b](https://huggingface.co/KRLabsOrg/squeez-2b)
- **Dataset:** [KRLabsOrg/tool-output-extraction-swebench](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)
- **Code:** [github.com/KRLabsOrg/squeez](https://github.com/KRLabsOrg/squeez)
- **Paper:** [arXiv (coming soon)]()

## References

- Hu, E. J., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. [ICLR](https://openreview.net/forum?id=nZeVKeeFYf9)
- Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)
- Jiang, H., et al. (2023). *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models*. [EMNLP](https://aclanthology.org/2023.emnlp-main.825/)
- Jiang, H., et al. (2024). *LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression*. [ACL](https://aclanthology.org/2024.acl-long.91/)
- Hwang, T., et al. (2025). *EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation*. [Findings of ACL](https://aclanthology.org/2025.findings-acl.253/)
- Chirkova, N., et al. (2025). *Provence: Efficient and Robust Context Pruning for Retrieval-Augmented Generation*. [arXiv](https://arxiv.org/abs/2501.16214)
- Zilliz. (2025). *Semantic Highlight Bilingual v1*. [Model card](https://huggingface.co/zilliz/semantic-highlight-bilingual-v1)
- Kerboua, I., et al. (2025). *FocusAgent: Simple Yet Effective Ways of Trimming the Large Context of Web Agents*. [arXiv](https://arxiv.org/abs/2510.03204)
- Wang, Y., et al. (2026). *SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents*. [arXiv](https://arxiv.org/abs/2601.16746)
- Jimenez, C. E., et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* [ICLR](https://openreview.net/forum?id=VTF8yNQM66)
- Yang, A., et al. (2025). *Qwen3 Technical Report*. [arXiv](https://arxiv.org/abs/2505.09388)
