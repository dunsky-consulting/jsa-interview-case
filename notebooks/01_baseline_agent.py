# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Jurisdictional scan assistant
# MAGIC
# MAGIC The agent lives in `src/jsa/agent.py`; its tools are the UC functions in `tools/`.
# MAGIC Every call is traced to the MLflow experiment below: open a trace to see each LLM
# MAGIC call, tool call and retrieved passage.
# MAGIC
# MAGIC Edit the code, re-run the cell that imports it (`importlib.reload`), and ask again.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import importlib
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

import mlflow

from jsa.config import settings_from_widgets

settings = settings_from_widgets(dbutils)
mlflow.set_experiment(settings.experiment_path)
mlflow.openai.autolog()

# COMMAND ----------

import jsa.agent
import jsa.tools

importlib.reload(jsa.tools)
importlib.reload(jsa.agent)
agent = jsa.agent.build_agent(settings)

# COMMAND ----------

dbutils.widgets.text("question", "What triggers a Distribution Deficiency Report under the AESO DDR Guideline?")
print(agent.answer(dbutils.widgets.get("question")))
