# Jurisdictional scan assistant: interview case

This is the code for Dunsky's Data & AI Engineer live case. Read it before your interview if you like. You can't run
it from here: the document library, search index, model endpoints and MLflow traces live in a Databricks workspace
you get access to when the interview starts.

## The interview (60 minutes)
- **Before:** read this code if you like (optional; nothing to install). You'll get an email invitation from
  Microsoft for the Databricks workspace: accept it ahead of time so you can sign in on the day.
- **In the interview** your access opens. We look at the five evaluation questions together; you predict which ones
  fail and why, then we check your predictions against the traces. You then spend about 20 minutes improving the
  agent and prove it with the evals. AI coding assistants are allowed; think out loud.
- **After** the interview your access is removed.

## The case
Our analysts run **jurisdictional scans**: how have utilities and regulators elsewhere handled AMI, DSM, grid
modernization, DER integration, load forecasting? A prototype assistant answers these questions from a library of
public filings. It demos fine on easy questions; in the interview you'll find out where it fails and why, improve it
live (AI coding assistants allowed), and prove the improvement with evals and traces.

The five evaluation questions, and the traces of the current agent failing them, are shown for the first time in the
interview. We'll ask you to predict which ones fail and why before we open the traces.

## What's here
| Path | What it is |
|---|---|
| `src/jsa/agent.py` | The agent: a plain OpenAI-SDK tool-calling loop. Start here. |
| `src/jsa/tools.py`, `tools/*.sql` | Its tools: Unity Catalog SQL functions `search_reports` and `get_report_metadata`. |
| `src/jsa/parsing.py`, `chunking.py`, `corpus.py`, `notebooks/00_setup.py`, `notebooks/00b_tools.py` | How the index and tools are built. |
| `evaluation/jsa_eval/` | The eval harness and scorers (deterministic checks plus MLflow LLM judges). |
| `notebooks/01_baseline_agent.py`, `notebooks/03_eval.py` | Ask the agent a question; run the eval. |

Stack: Azure Databricks (Unity Catalog, Vector Search, Model Serving), an open-weight LLM (gpt-oss-120b), MLflow 3
tracing and `mlflow.genai.evaluate`, plain Python.
