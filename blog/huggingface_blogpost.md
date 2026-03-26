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

We trained and open-sourced **Squeez-2B**, a small model that prunes tool output for coding agents. It keeps the lines that matter and drops the rest, achieving 0.86 recall at 91.5% compression. The model is a LoRA-tuned Qwen 3.5 2B, served via vLLM or used as a CLI pipe.

**Release:**

- Model: [KRLabsOrg/squeez-2b](https://huggingface.co/KRLabsOrg/squeez-2b) (Apache 2.0)
- Dataset: [KRLabsOrg/tool-output-extraction-swebench](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)
- Code + CLI: [github.com/KRLabsOrg/squeez](https://github.com/KRLabsOrg/squeez)

In this post we share our approach: what problem we are solving, how we built the benchmark, and what works.

## The Problem: Agents Waste Most of Their Context on Noise

Coding agents such as Claude Code and Codex spend much of their time reading tool output. When an agent runs `pytest`, `grep`, `git log`, or `kubectl`, the result is often hundreds of lines. But only a handful of those lines are relevant for the next step. The rest is boilerplate: passing tests, headers, unchanged files, timestamps.

Here is what that looks like in practice. An agent runs `pytest` and gets 45 lines back. Only 6 matter:

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

That is **87% compression** while keeping everything the agent needs for the next debugging step. The command is:

```bash
python -m pytest tests/ -v 2>&1 | squeez "find the test failure related to authentication"
```

This problem exists across all tool types: `grep` results buried in hundreds of matches, `git log` with dozens of commits where only one matters, `kubectl describe` with 250 lines of pod metadata when the agent needs the two-line OOMKilled block. The common thread is that the useful evidence is a small region inside a much larger observation.

## Why Existing Pruning Tools Don't Solve This

We looked at several existing systems before building our own.

**LLMLingua / LongLLMLingua** ([Jiang et al., 2023](https://aclanthology.org/2023.emnlp-main.825/); [Jiang et al., 2024](https://aclanthology.org/2024.acl-long.91/)) compress prompts by removing tokens or prompt blocks. They work at the token level on natural language. Tool output is different: a single removed token in a stack trace can break the meaning of the entire block.

**EXIT** ([Hwang et al., 2025](https://aclanthology.org/2025.findings-acl.253/)) performs extractive compression on retrieved documents for question answering. It assumes well-formed prose with sentence boundaries. Tool output does not have sentence boundaries. A pytest traceback, a `git blame` output, and a Terraform plan are not prose.

**Provence** ([Chirkova et al., 2025](https://arxiv.org/abs/2501.16214)) and **Zilliz Semantic Highlight** ([model card](https://huggingface.co/zilliz/semantic-highlight-bilingual-v1)) push context pruning further with sequence labeling and token-level scoring. They achieve strong results on retrieved documents. But they are designed for passage text, not the mixed format of tool output where code, logs, stack traces, metadata, and structured output are interleaved in a single artifact.

**SWE-Pruner** ([Wang et al., 2026](https://arxiv.org/abs/2601.16746)) is the closest to our setting. It targets coding agents specifically. However, it focuses on pruning repository code context rather than individual tool observations. An agent's tool output is not just source code: it includes shell output, build logs, test results, Git history, and package-manager traces.

No existing system handles the full range of mixed-format tool output that coding agents produce. That is the gap we set out to fill.

## The Task

We formulate the problem as follows: given a focused query and one raw tool observation, return the smallest verbatim evidence block that the agent should inspect next. The output is therefore not a summary or a rewrite. It is a subset of the source lines. The query is also not the full issue description, but a narrower local need such as finding the failure block, the relevant code region, or the commit entry that matters for the next debugging step.

The pipeline from raw tool output to trained model is shown below:

<p align="center">
  <img src="./assets/squeez_overview.svg" alt="Squeez pipeline: from raw tool output through span annotation to generative model" width="920">
</p>

## Building the Benchmark

### Data Sources

The benchmark is built from two sources.

The first is [SWE-bench](https://openreview.net/forum?id=VTF8yNQM66), a collection of real GitHub issue-resolution tasks. We do not use it as a patch-generation benchmark. Instead, we use it as a source of realistic repository snapshots and tool observations. Starting from cloned SWE-bench repositories, we collected **10,713** raw tool observations: file reads, grep hits, Git history, shell output, test results, and package-manager logs.

The second source is synthetic multi-ecosystem tool output covering TypeScript, Go, Rust, Java, Docker, Terraform, and Kubernetes. Its role is to extend coverage where SWE-bench is thin. We begin from **2,039** raw synthetic observations and also construct explicit negatives (cases where the correct pruning decision is to return nothing).

### Annotation with a Teacher Model

Each example is built in two stages with `openai/gpt-oss-120b` as the teacher. First, the teacher writes a focused extraction query for one observation. This is not the full issue description but a narrower need: "find the failure block," "find the relevant code region," "find the commit entry that matters." Second, the teacher selects the smallest contiguous span that answers that query.

The key design decision: every target is a verbatim subset of the source. The teacher sees a numbered view of the output for stable span selection, but the released labels are mapped back onto the original raw text. This makes it a pruning benchmark, not a summarization benchmark.

### Test Set Curation

We manually curated the held-out test set. Starting from 729 candidates, we removed 111 (15.2%) that were near-duplicates, trivial outputs, overly broad spans, or incorrect annotations. The final test set contains **618** manually reviewed examples.

### Dataset Statistics

The released benchmark contains **11,477** examples: 9,205 SWE-derived, 1,697 synthetic positives, and 575 synthetic negatives, covering **27 tool types**. We split SWE-derived examples by repository and synthetic examples by tool family.

The source breakdown is:

| Source | Raw inputs | Released rows |
|---|---:|---:|
| SWE-derived | 10,713 | 9,205 |
| Synthetic positives | 2,039 | 1,697 |
| Synthetic negatives | — | 575 |
| Total | 12,752 | 11,477 |

The table below shows the largest tool families. The distribution is deliberately varied: `python` errors average 60 tokens while `type_check` outputs average 3,400 and `git_blame` over 4,200. This is one reason heuristic baselines fail: the relevant evidence does not follow one structural pattern.

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

## Model and Training

We chose **Qwen 3.5 2B** ([Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)) as the base model. The goal is not to maximize zero-shot reasoning with the largest possible model, but to learn a narrow extraction policy that runs cheaply inside agent systems. A 2B model is a good fit: strong enough to benefit from fine-tuning, small enough to serve on a single GPU.

We fine-tuned with **LoRA** ([Hu et al., 2022](https://openreview.net/forum?id=nZeVKeeFYf9); [Dettmers et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)) using the **Unsloth** stack. The model receives a focused query and a raw tool observation, and is trained to emit the extracted evidence wrapped in `<relevant_lines>` tags. Training runs for 3 epochs with effective batch size 32 and max sequence length 20,000. The final model is merged and served through **vLLM**.

## Results

We compared Squeez-2B against three zero-shot models and four heuristic baselines. The heuristic baselines keep approximately 10% of input lines to match the typical gold ratio.

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

**Task-specific training matters.** A fine-tuned 2B model outperforms the 18x larger Qwen 3.5 35B A3B by 11.3 recall points at essentially the same compression level. The untrained 2B base, given the same prompt, reaches only 0.53 recall. Fine-tuning nearly doubles it.

**Heuristics are not sufficient.** BM25 reaches only 0.22 recall on tool output. Relevant lines may appear at the end of a traceback, in the middle of a build log, or inside a short code block with no lexical overlap with the query.

**Aggressive compression alone is not enough.** Kimi K2 removes the most tokens (94.3%) but pays for it in recall (0.53 vs 0.86).

The recall-compression trade-off is shown below. Squeez-2B occupies the upper-left: high recall with strong compression.

<p align="center">
  <img src="./assets/squeez_results_chart.svg" alt="Recall vs compression across all models" width="920">
</p>

## What the Model Learns

The model appears to learn tool-specific regularities:

- In `grep` and `git_log`, it returns the single relevant hit instead of a broader keyword neighborhood
- In `test_output` and `build_output`, it keeps the failure block and drops the boilerplate
- In `read_file`, it retains the smallest contiguous code block that answers the query

Here is an example from a `kubectl` observation. The full output is 250 lines of pod description. The relevant evidence is two lines:

<p align="center">
  <img src="./assets/squeez_qualitative_example.svg" alt="kubectl example: 2 relevant lines from 250" width="920">
</p>

The strongest remaining failures are semantically adjacent but wrong: choosing the wrong file from an `ls` listing, or returning a related commit that touches the same module but does not answer the query.

## Using Squeez in Your Agent

Squeez is a preprocessing step. It does not change the agent's planner, tool API, or interaction loop.

As a CLI pipe:

```bash
pytest -q 2>&1 | squeez "find the failure block"
git log --oneline -50 | squeez "find the commit that changed CSRF handling"
cat src/auth/middleware.py | squeez "find the referer validation logic"
```

With vLLM for production:

```bash
vllm serve KRLabsOrg/squeez-2b --dtype bfloat16 --max-model-len 16384
export SQUEEZ_SERVER_URL=http://localhost:8000/v1
pytest -q 2>&1 | squeez "find the failure block"
```

To add it to **Claude Code**, put this in your `CLAUDE.md`:

```
When you invoke a shell command, pipe it through `squeez` and describe what you need.
Examples:
- `bun test 2>&1 | squeez "did the tests pass?"`
- `git log --oneline -50 | squeez "find the commit that broke CSRF"`
```

The same pattern works with Codex, OpenHands, SWE-agent, or any agent that accepts system-level instructions.

Our contribution is intentionally narrow. We do not propose a new general-purpose prompt compressor, and we do not claim to solve end-to-end software engineering. Instead, we isolate one recurring bottleneck in coding agents: deciding what to keep from a single tool observation before the next reasoning step. The benchmark, the released model, and the results all point in the same direction: this narrow problem is learnable, practically useful, and not handled well by simple heuristics or larger zero-shot models alone.

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
- Yang, J., et al. (2024). *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*. [NeurIPS](https://arxiv.org/abs/2405.15793)
- Wang, X., et al. (2024). *OpenHands: An Open Platform for AI Software Developers as Generalist Agents*. [arXiv](https://arxiv.org/abs/2407.16741)
- See, A., Liu, P. J., and Manning, C. D. (2017). *Get To The Point: Summarization with Pointer-Generator Networks*. [ACL](https://aclanthology.org/P17-1099/)
- Liu, Y. and Lapata, M. (2019). *Text Summarization with Pretrained Encoders*. [EMNLP](https://aclanthology.org/D19-1387/)
- Rajpurkar, P., et al. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. [EMNLP](https://aclanthology.org/D16-1264/)
- Thorne, J., et al. (2018). *FEVER: A Large-scale Dataset for Fact Extraction and VERification*. [NAACL](https://aclanthology.org/N18-1074/)
- Yang, Z., et al. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. [EMNLP](https://aclanthology.org/D18-1259/)
- Yang, A., et al. (2025). *Qwen3 Technical Report*. [arXiv](https://arxiv.org/abs/2505.09388)
