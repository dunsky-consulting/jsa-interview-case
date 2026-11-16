# Databricks notebook source
# MAGIC %md
# MAGIC # Hello agent: tool calling, tracing and evaluation on toy data
# MAGIC
# MAGIC A ~40-line agent over three made-up documents, to get familiar with:
# MAGIC 1. a plain OpenAI-SDK tool-calling loop against a Databricks serving endpoint,
# MAGIC 2. MLflow 3 traces (open the experiment after running and click a trace),
# MAGIC 3. `mlflow.genai.evaluate` with a custom deterministic scorer and a built-in LLM judge.

# COMMAND ----------

# MAGIC %pip install -q "mlflow[databricks]>=3.17,<4" "openai>=1.100,<3"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.genai.scorers import ExpectationsGuidelines, scorer

dbutils.widgets.text("llm_endpoint", "jsa-gpt-oss")
dbutils.widgets.text("experiment_path", "/Workspace/jsa-interview/<your interview ID>/experiment")
ENDPOINT = dbutils.widgets.get("llm_endpoint")
mlflow.set_experiment(dbutils.widgets.get("experiment_path"))
mlflow.openai.autolog()
client = WorkspaceClient().serving_endpoints.get_open_ai_client()

DOCS = {
    "bakery_hours": "The Riverside Bakery opens at 7:00 on weekdays and 8:30 on weekends.",
    "bakery_menu": "Riverside Bakery sells sourdough ($9), croissants ($4) and seasonal pies ($24).",
    "library_hours": "The Elm Street Library is open 10:00-20:00, closed on Mondays.",
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Keyword search over local documents; returns matching documents with their ids.",
        "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]},
    },
}]

# COMMAND ----------

@mlflow.trace(span_type=SpanType.RETRIEVER)
def lookup(keyword: str) -> list[dict]:
    return [{"page_content": text, "metadata": {"doc_id": doc_id}}
            for doc_id, text in DOCS.items() if keyword.lower() in text.lower()]


@mlflow.trace(span_type=SpanType.AGENT)
def answer(question: str) -> str:
    messages = [
        {"role": "system", "content": "Answer questions using the lookup tool."},
        {"role": "user", "content": question},
    ]
    while True:
        message = client.chat.completions.create(model=ENDPOINT, messages=messages, tools=TOOLS).choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return message.content if isinstance(message.content, str) else str(message.content)
        for call in message.tool_calls:
            results = lookup(**json.loads(call.function.arguments))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(results)})


print(answer("When does the bakery open on Saturday?"))

# COMMAND ----------

@scorer
def retrieved_something(trace) -> bool:
    """Deterministic scorers can read the trace: here, did any RETRIEVER span return a document?"""
    return any(span.outputs for span in trace.search_spans(span_type=SpanType.RETRIEVER))


data = [
    {"inputs": {"question": "When does the bakery open on Saturday?"},
     "expectations": {"guidelines": ["States that the bakery opens at 8:30 on Saturday"]}},
    {"inputs": {"question": "How much is a croissant?"},
     "expectations": {"guidelines": ["States that a croissant costs $4"]}},
    {"inputs": {"question": "Which day is the library closed?"},
     "expectations": {"guidelines": ["States that the library is closed on Mondays"]}},
]
result = mlflow.genai.evaluate(
    data=data, predict_fn=answer, scorers=[retrieved_something, ExpectationsGuidelines(model=f"databricks:/{ENDPOINT}")]
)
display(result.metrics)
