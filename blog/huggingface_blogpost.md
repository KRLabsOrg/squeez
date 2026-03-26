---
title: "Squeez: Task-Conditioned Tool-Output Pruning for Coding Agents"
emoji: "🗜️"
colorFrom: "indigo"
colorTo: "blue"
sdk: "static"
pinned: false
license: "mit"
---

# Squeez: Task-Conditioned Tool-Output Pruning for Coding Agents

<p align="center">
  <img src="./assets/squeez_mascot.png" alt="Squeez mascot" width="180">
</p>

## TL;DR

- **Problem:** coding agents often spend most of their context budget on irrelevant tool output
- **Solution:** task-conditioned pruning of single tool observations, trained on 11,477 examples across 27 tool types
- **Model:** LoRA-tuned Qwen 3.5 2B, served with vLLM or as a CLI pipe
- **Result:** 0.86 recall at 91.5% compression, +11.3 recall points over zero-shot Qwen 3.5 35B A3B
- **Links:** [Model](https://huggingface.co/KRLabsOrg/squeez-2b) · [Dataset](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench) · [Code](https://github.com/KRLabsOrg/squeez)

## The problem

Coding agents work over streams of file reads, grep hits, stack traces, build logs, API responses, and version-control history ([SWE-agent](https://arxiv.org/abs/2405.15793), [OpenHands](https://arxiv.org/abs/2407.16741)). In practice, most of that output is noise. Here is a real example — an agent runs `pytest` and gets 45 lines back, but only 6 lines matter:

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

This is **87% compression** while preserving the only part of the test run that matters for the next debugging step.

```bash
python -m pytest tests/ -v 2>&1 | squeez "find the test failure related to authentication"
```

Recent pruning systems operate on different units in different settings. **LLMLingua** and **LongLLMLingua** compress prompts at the token or prompt-block level ([Jiang et al., 2023](https://aclanthology.org/2023.emnlp-main.825/); [Jiang et al., 2024](https://aclanthology.org/2024.acl-long.91/)). **EXIT** performs extractive compression over retrieved documents for downstream question answering ([Hwang et al., 2025](https://aclanthology.org/2025.findings-acl.253/)). **Provence** formulates context pruning as sequence labeling and combines pruning with reranking for retrieval-augmented generation ([Chirkova et al., 2025](https://arxiv.org/abs/2501.16214)). **Zilliz Semantic Highlight** adapts this line to lightweight semantic highlighting by scoring token and sentence relevance in retrieved text ([model card](https://huggingface.co/zilliz/semantic-highlight-bilingual-v1)). **SWE-Pruner** is closer to our setting, but focuses on repository code pruning rather than mixed single-tool observations ([Wang et al., 2026](https://arxiv.org/abs/2601.16746)). Tool output is different: it mixes code, logs, shell traces, stack frames, JSON payloads, and Git metadata in unpredictable ways. That is the gap Squeez targets.

## The task

We formulate this as **task-conditioned tool-output pruning**: given a focused query $q$ and one raw tool observation $o$, return the smallest verbatim evidence block that the agent should inspect next.

This is intentionally narrower than bug solving. The model is not asked to infer the correct patch from one observation. It is asked to preserve the relevant evidence while removing the rest. The output must be a verbatim subset of the input — no rewriting, no summarization.

That last point matters. Tool output often contains details that should not be paraphrased away: a concrete exit code, a failing import path, a version mismatch, the one function body that controls the relevant behavior, or the single Git commit that touched the relevant module. In that sense, the task is closer to evidence-grounded selection than to abstractive summarization ([See et al., 2017](https://aclanthology.org/P17-1099/); [Liu and Lapata, 2019](https://aclanthology.org/D19-1387/)), and shares the faithfulness concern of extractive QA ([SQuAD](https://aclanthology.org/D16-1264/), [FEVER](https://aclanthology.org/N18-1074/), [HotpotQA](https://aclanthology.org/D18-1259/)).

The following diagram shows the full pipeline from raw tool output through grounded span annotation to the generative training format and evaluation:

<p align="center">
  <img src="./assets/squeez_overview.svg" alt="Squeez pipeline: from raw tool output through grounded span annotation to generative model and evaluation" width="920">
</p>

## The benchmark

The released benchmark contains **11,477** examples: **9,205** SWE-derived examples built from repository interactions on [SWE-bench](https://openreview.net/forum?id=VTF8yNQM66), and **2,272** synthetic examples (1,697 positives and 575 explicit negatives). The synthetic portion extends coverage beyond the Python-heavy SWE setting to TypeScript, Go, Rust, Java, Docker, Terraform, and Kubernetes workflows. In total, the benchmark covers **27 tool types**, including `read_file`, `grep`, `git_log`, `git_blame`, `git_diff`, `ls`, `curl`, `python`, `test_output`, `build_output`, `pip_install`, `type_check`, `kubectl`, `docker_build`, `cargo_build`, `terraform`, and others.

Labels are produced with `openai/gpt-oss-120b`. For each raw tool output, the teacher writes a focused extraction query and selects one or more contiguous answer spans. The released labels are mapped back onto the original raw text, so every target remains a verbatim subset of the source. For the generative model, these gold spans are linearized as XML-wrapped extracted text.

We manually curated the held-out test set. Starting from 729 candidate test examples, we removed 111 cases (15.2%) that were near-duplicates, trivial 1-2 line outputs, overly broad spans, or incorrect annotations. The final test set contains **618** manually reviewed examples.

The table below shows the largest tool families. The benchmark mixes very short observations (e.g., `python` errors averaging 59 tokens) with long, structurally noisy ones (e.g., `type_check` outputs averaging 3,428 tokens). This variation is one reason simple truncation and lexical retrieval behave poorly: the relevant evidence does not follow one structural pattern, and it may occur at the beginning, middle, or end of the observation.

| Tool family | Rows | Avg. input tokens | Avg. gold tokens |
|---|---:|---:|---:|
| `read_file` | 3760 | 1680 | 84 |
| `grep` | 1320 | 785 | 19 |
| `git_log` | 720 | 161 | 11 |
| `python` | 670 | 59 | 27 |
| `curl` | 488 | 726 | 68 |
| `pip_install` | 441 | 438 | 79 |
| `type_check` | 316 | 3428 | 39 |
| `tsc` | 229 | 1444 | 56 |
| other 19 tools | 3533 | 926 | 52 |

## Model and training

The released model, **Squeez-2B**, is [Qwen 3.5 2B](https://arxiv.org/abs/2505.09388) fine-tuned with LoRA ([Hu et al., 2022](https://openreview.net/forum?id=nZeVKeeFYf9); [Dettmers et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)). We chose Qwen 3.5 2B because it offers a strong trade-off between base capability, fine-tuning cost, and deployment simplicity. The point here is not to maximize zero-shot reasoning depth with the largest possible generator, but to learn a narrow supervised extraction policy that can run cheaply inside agent systems. A 2B open model is therefore a better fit for the intended use case than a much larger decoder, while still remaining strong enough to benefit from task-specific supervision.

The model receives a focused extraction query and the raw tool observation, and is trained to emit the extracted evidence block wrapped in `<relevant_lines>` tags. Training uses the Unsloth stack with LoRA, maximum sequence length 20,000, effective batch size 32 (8 per device, 4 gradient accumulation), learning rate 2e-4, 3 epochs, warmup 0.05, weight decay 0.01. The final model is merged and served through vLLM.

## Results

We compare Squeez-2B against three zero-shot generative baselines (Qwen 3.5 35B A3B, Kimi K2, and the untrained Qwen 3.5 2B base) and four heuristics (BM25, First-N, Last-N, Random) that each keep approximately 10% of input lines. Evaluation focuses on **recall under strong compression**, with F1 as the summary metric. Dropping relevant evidence is usually more harmful than keeping a slightly larger block, which is why recall matters more than precision for this task.

| Model | Recall | F1 | Compression |
|---|---:|---:|---:|
| **Squeez-2B** | **0.8624** | **0.8035** | 0.9150 |
| Qwen 3.5 35B A3B | 0.7498 | 0.7254 | 0.9177 |
| Kimi K2 | 0.5286 | 0.6827 | **0.9425** |
| Qwen 3.5 2B (base) | 0.5299 | 0.5482 | 0.8197 |
| BM25 (10%) | 0.2172 | 0.2314 | 0.9036 |
| First-N (10%) | 0.1445 | 0.1570 | 0.9055 |
| Random (10%) | 0.1009 | 0.1966 | 0.9067 |
| Last-N (10%) | 0.0503 | 0.1393 | 0.9130 |

Three things stand out.

**Task-specific training matters.** A fine-tuned 2B model outperforms a much larger 35B zero-shot model by 11.3 recall points while operating at essentially the same compression level. The untrained 2B base model, given the same prompt, reaches only 0.53 recall — fine-tuning nearly doubles it.

**Generic heuristics are not sufficient.** BM25, which works well for document retrieval, reaches only 0.22 recall on tool output. Relevant lines may appear at the end of a traceback, in the middle of a build log, or inside a short code block that is only weakly signaled lexically by the query.

**Aggressive compression alone is not enough.** Kimi K2 removes the largest fraction of tokens (94.3%), but the gain in compression comes with a substantial recall penalty (0.53 vs 0.86).

The recall-compression trade-off across all models is shown below. Squeez-2B occupies the upper-left region: high recall with strong compression. The heuristic baselines cluster in the lower-right, achieving similar compression but recovering almost none of the relevant evidence.

<p align="center">
  <img src="./assets/squeez_results_chart.svg" alt="Recall vs compression trade-off across all models and baselines" width="920">
</p>

## What the model learns

The fine-tuned model appears to learn tool-specific pruning regularities. In `grep` and `git_log` outputs, it returns the single relevant hit rather than a broader keyword-matching neighborhood. In `test_output`, `build_output`, and package-manager logs, it keeps the actual failure block instead of surrounding boilerplate. In `read_file`, it retains the smallest contiguous code block that answers the query rather than an entire surrounding function or class.

The following example shows a `kubectl` observation where the relevant evidence is a two-line block buried in 250 lines of pod description. The model learns to extract just the OOMKilled reason and exit code:

<p align="center">
  <img src="./assets/squeez_qualitative_example.svg" alt="Qualitative example: kubectl output where Squeez extracts the two relevant lines from 250 lines of pod description" width="920">
</p>

The strongest remaining failures are semantically adjacent but incorrect selections: choosing the wrong file from an `ls` listing, or returning a related commit that touches the same module but does not directly answer the query.

## Using Squeez in your agent

Squeez is designed as a drop-in preprocessing step for existing coding agents. It does not require changes to the agent's planner, tool API, or interaction loop.

As a CLI pipe:

```bash
pytest -q 2>&1 | squeez "find the failure block"
git log --oneline -50 | squeez "find the commit that changed CSRF handling"
cat src/auth/middleware.py | squeez "find the referer validation logic"
```

With vLLM for production throughput:

```bash
vllm serve KRLabsOrg/squeez-2b --dtype bfloat16 --max-model-len 16384
export SQUEEZ_SERVER_URL=http://localhost:8000/v1
pytest -q 2>&1 | squeez "find the failure block"
```

To add Squeez to **Claude Code**, a minimal `CLAUDE.md` instruction is enough:

```
When you invoke a shell command, pipe it through `squeez` and describe what you need.
Examples:
- `bun test 2>&1 | squeez "did the tests pass?"`
- `git log --oneline -50 | squeez "find the commit that broke CSRF"`
```

The same pattern works with Codex, OpenHands, SWE-agent, or any other agent that accepts system-level instructions or shell wrappers.

## Discussion

Coding-agent systems increasingly bottleneck on context rather than on the ability to produce plausible next actions. They can often issue the right tools, but they still have to decide what to keep from the resulting observations. That decision is frequently treated as an implementation detail, handled with truncation, heuristics, or manual prompt engineering.

We think it deserves to be studied as a first-class learning problem. Squeez isolates a small but practically important part of the coding-agent pipeline: one focused query, one raw tool observation, one verbatim evidence block. This formulation is simpler than full end-to-end agent evaluation, but rich enough to matter in real coding-agent loops.

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
