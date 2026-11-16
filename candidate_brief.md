# Candidate brief: Jurisdictional scan assistant

## Context
Our analysts run **jurisdictional scans**: how have utilities and regulators elsewhere handled AMI,
DSM, grid modernization, DER integration, load forecasting? A colleague built a prototype assistant
that answers these questions from a library of public filings: utility applications, regulator
decisions, strategic plans and research reports from Alberta, Hawaii, Québec, New York, California
and others.

It demos fine on easy questions. Before more people rely on it, we want to know where it fails and
we want it to get better, **with evidence**.

Your access to this workspace starts with the interview. We'll first walk through the code and the seeded traces
together: how the system is built, which questions fail and why.

## Your task (≈20 minutes, then we re-run the evals together)
1. **Diagnose.** Walk us through the seeded traces: what is going wrong, and why?
2. **Improve** the agent. Prioritize: you won't fix everything, and choosing what to fix is part of the exercise.
3. **Prove it.** Run the visible eval and compare it with the seeded baseline traces, per question and per scorer.

AI coding assistants are allowed. Think out loud: we care about your reasoning more than your typing speed.

## What's in the workspace
| Path | What it is |
|---|---|
| `src/jsa/agent.py` | The agent: a plain OpenAI-SDK tool-calling loop. Start here. |
| `src/jsa/tools.py`, `tools/*.sql` | Tools: Unity Catalog functions `search_reports` and `get_report_metadata`. |
| `src/jsa/chunking.py`, `notebooks/00_setup.py`, `notebooks/00b_tools.py` | How the index and tools were built. Rebuilding takes ~15 min; write to your sandbox schema `jsa_sandbox_<your interview ID>` if you try. |
| `config/corpus_manifest.yml` | The library: one entry per document, with metadata. |
| `notebooks/01_baseline_agent.py` | Ask the agent a question interactively; every call is traced. |
| `notebooks/03_eval.py` | Run the visible eval (`eval/visible.jsonl`) with all scorers. |
| MLflow experiment `experiment` (in your folder) | Seeded traces of failing baseline runs (tag `seed=baseline_failure`) plus your runs. |

The workspace, data, embeddings and index are in Canada Central. The agent's LLM is an open-weight model
(gpt-oss-120b, endpoint `jsa-gpt-oss`); the LLM judges use a different model (gpt-5.4-mini, endpoint `jsa-judge`)
so the agent doesn't grade itself. Both are served from Azure AI Foundry and their processing may happen outside
Canada, which is acceptable here because the library is public. Don't call other external model APIs.

## Contract (so the eval can score your agent)
- `jsa.agent.build_agent(settings, client=None, backend=None)` returns an object with `answer(question) -> str`.
  The harness passes `client` (it caps LLM calls per question) and `backend`, so use them when they're given.
- Retrieval is scored from **RETRIEVER spans** whose outputs carry `metadata.doc_id` (`jsa.tools.as_retriever_documents`).
- Citations are recognized in the form **`[doc_id, p. N]`**.
- When the library does not contain the answer, the agent should **say so explicitly** rather than guess.

## Scoring dimensions
| Scorer | Kind |
|---|---|
| `retrieval_hit`: an expected document was retrieved | deterministic |
| `correct_not_found`: declines when it should, answers when it should | deterministic |
| `citation_present`: cites a real document and page | deterministic |
| `expectations_guidelines`: the answer meets the question's guidelines in the eval set | LLM judge |
| `retrieval_groundedness`: the answer is supported by the retrieved passages | LLM judge |
| `relevance_to_query`: the answer addresses the question asked | LLM judge |
| `tool_call_efficiency`: no redundant or looping tool calls | LLM judge |
| `safety`: no harmful content | LLM judge |
| `latency_seconds`, `tokens`, `steps` | deterministic |

A question **passes** when `correct_not_found` and `expectations_guidelines` both pass; the other scorers are diagnostics.

Please don't edit `evaluation/`: scores must stay comparable across runs and candidates.
