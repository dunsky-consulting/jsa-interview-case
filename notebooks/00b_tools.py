# Databricks notebook source
# MAGIC %md
# MAGIC # 00b · Tools: UC functions
# MAGIC
# MAGIC Creates the agent's tools from `tools/*.sql`:
# MAGIC - `search_reports(query, k, utility, doc_type, jurisdiction)`: semantic search over the index
# MAGIC - `get_report_metadata(doc_id)`: document metadata (NULL lists the library)
# MAGIC
# MAGIC Runs on the SQL warehouse (vector_search() is not available on classic compute).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

from jsa.config import settings_from_widgets
from jsa.tools import TOOL_FUNCTIONS, UCFunctionBackend, render_tool_sql

settings = settings_from_widgets(dbutils)
w = WorkspaceClient()

# COMMAND ----------

for function in TOOL_FUNCTIONS:
    response = w.statement_execution.execute_statement(
        statement=render_tool_sql(function, settings), warehouse_id=settings.warehouse_id, wait_timeout="50s"
    )
    if response.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"{function}: {response.status.error.message if response.status.error else response.status.state}")
    print(f"created {settings.fqn(function)}")

# COMMAND ----------

# MAGIC %md Smoke test both tools through the same client the agent uses.

# COMMAND ----------

backend = UCFunctionBackend(settings, w)
hits = backend.search_reports("advanced metering infrastructure business case", k=3)
assert hits, "search_reports returned nothing: check the index and embedding endpoint"
display(hits)
display(backend.get_report_metadata(hits[0]["doc_id"]))
