# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Evaluate the agent
# MAGIC
# MAGIC Runs `mlflow.genai.evaluate` over an eval set and logs a run to the experiment.
# MAGIC
# MAGIC | Scorer | Kind | What it checks |
# MAGIC |---|---|---|
# MAGIC | `retrieval_hit` | deterministic | an expected document appears in a RETRIEVER span (answerable questions only) |
# MAGIC | `correct_not_found` | deterministic | declines when the library lacks the answer, answers when it has it |
# MAGIC | `citation_present` | deterministic | at least one `[doc_id, p. N]` citation to a real document |
# MAGIC | `expectations_guidelines` | LLM judge (guidelines) | the answer meets the question's guidelines in the eval set |
# MAGIC | `retrieval_groundedness` | LLM judge (RAG) | the answer is supported by the retrieved passages |
# MAGIC | `relevance_to_query` | LLM judge (response quality) | the answer addresses the question asked |
# MAGIC | `tool_call_efficiency` | LLM judge (tool calls) | no redundant or looping tool calls |
# MAGIC | `safety` | LLM judge (safety) | no harmful content |
# MAGIC | `latency_seconds`, `tokens`, `steps` | deterministic | cost and effort per question |
# MAGIC
# MAGIC A question **passes** when `correct_not_found` and `expectations_guidelines` both pass; the other scorers are diagnostics.
# MAGIC
# MAGIC Parameters: `eval_set` (JSONL path), `agent_module` (module exposing `build_agent`),
# MAGIC `agent_src` (directory put first on `sys.path` for the agent code).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

dbutils.widgets.text("agent_src", "../src")
dbutils.widgets.text("agent_module", "jsa.agent")
dbutils.widgets.text("eval_set", "../eval/visible.jsonl")
dbutils.widgets.dropdown("include_judge", "true", ["true", "false"])

# The agent package comes from agent_src; the scorers always come from this deployment's evaluation/.
sys.path[:0] = [os.path.abspath(dbutils.widgets.get("agent_src")), os.path.abspath("../evaluation")]

import mlflow

from jsa.config import settings_from_widgets
from jsa_eval.calibration import results_table
from jsa_eval.evalsets import load_eval_set
from jsa_eval.harness import run_eval

settings = settings_from_widgets(dbutils)
mlflow.set_experiment(settings.experiment_path)
mlflow.openai.autolog()

# COMMAND ----------

eval_set = dbutils.widgets.get("eval_set")
cases = load_eval_set(eval_set)
result = run_eval(
    cases,
    settings,
    agent_module=dbutils.widgets.get("agent_module"),
    include_judge=dbutils.widgets.get("include_judge") == "true",
    run_name=f"eval {os.path.basename(eval_set)}",
)
display(result.metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results per question
# MAGIC `result` is the pass rule; `why` is the guidelines judge's explanation. Open a `trace_id` in the
# MAGIC experiment's Traces tab to see every step and every judge's reasoning.

# COMMAND ----------

display(results_table(result.result_df, cases))
