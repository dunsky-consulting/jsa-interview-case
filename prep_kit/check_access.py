# Databricks notebook source
# MAGIC %md
# MAGIC # Access check
# MAGIC Fill the widgets with the values we sent you, then **Run all**. Each cell prints OK or the error.

# COMMAND ----------

# MAGIC %pip install -q "mlflow[databricks]>=3.17,<4" "openai>=1.100,<3"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("llm_endpoint", "jsa-gpt-oss")
dbutils.widgets.text("warehouse_id", "")
dbutils.widgets.text("experiment_path", "/Workspace/jsa-interview/<your interview ID>/experiment")

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
print("OK workspace:", w.config.host, "user:", w.current_user.me().user_name)

# COMMAND ----------

client = w.serving_endpoints.get_open_ai_client()
reply = client.chat.completions.create(
    model=dbutils.widgets.get("llm_endpoint"),
    messages=[{"role": "user", "content": "Reply with the single word: ready"}],
    max_tokens=50,
)
print("OK LLM:", reply.choices[0].message.content)

# COMMAND ----------

result = w.statement_execution.execute_statement(
    statement="SELECT current_user() AS me", warehouse_id=dbutils.widgets.get("warehouse_id"), wait_timeout="30s"
)
print("OK SQL warehouse:", result.status.state, result.result.data_array if result.result else None)

# COMMAND ----------

import mlflow

experiment = mlflow.set_experiment(dbutils.widgets.get("experiment_path"))
print("OK experiment:", experiment.experiment_id)
