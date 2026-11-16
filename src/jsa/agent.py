"""Jurisdictional scan assistant: a tool-calling agent over the report library.

Contract used by the eval notebooks: this module exposes ``build_agent(settings)``
returning an object with ``answer(question: str) -> str``.
"""

import json
import traceback

import mlflow
from mlflow.entities import SpanType

from jsa.config import Settings
from jsa.tools import ToolBackend, UCFunctionBackend, as_retriever_documents

SYSTEM_PROMPT = (
    "You are a regulatory analyst assistant specialized in Alberta cost-of-service (COS) filings. "
    "Use the search_reports tool to find relevant passages, then answer the user's question."
)

UTILITIES = ["ATCO Electric", "FortisAlberta", "ENMAX", "EPCOR"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_reports",
            "description": "Search Alberta COS filings for passages relevant to the query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "utility": {"type": "string", "enum": UTILITIES, "description": "Filter to one utility."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_report_metadata",
            "description": "Get metadata for a report.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
            },
        },
    },
]


def message_text(content) -> str:
    """Chat content may be a string or a list of typed parts (reasoning models); keep the text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")


class Agent:
    def __init__(self, client, model: str, backend: ToolBackend):
        self.client = client
        self.model = model
        self.backend = backend

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def search(self, query: str, utility: str | None = None) -> list[dict]:
        return as_retriever_documents(self.backend.search_reports(query=query, k=3, utility=utility))

    @mlflow.trace(span_type=SpanType.TOOL)
    def run_tool(self, name: str, arguments: dict) -> str:
        try:
            if name == "search_reports":
                documents = self.search(arguments["query"], arguments.get("utility"))
                return "\n\n".join(d["page_content"] for d in documents)
            if name == "get_report_metadata":
                return json.dumps(self.backend.get_report_metadata(arguments["doc_id"]))
            raise ValueError(f"Unknown tool {name}")
        except Exception:
            return traceback.format_exc()

    @mlflow.trace(span_type=SpanType.AGENT)
    def answer(self, question: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
        while True:
            response = self.client.chat.completions.create(model=self.model, messages=messages, tools=TOOLS)
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                return message_text(message.content)
            for call in message.tool_calls:
                result = self.run_tool(call.function.name, json.loads(call.function.arguments))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})


def build_agent(settings: Settings, client=None, backend: ToolBackend | None = None) -> Agent:
    """Wire the agent to the workspace LLM endpoint and UC function tools (overridable for tests)."""
    if client is None or backend is None:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        client = client or w.serving_endpoints.get_open_ai_client()
        backend = backend or UCFunctionBackend(settings, w)
    return Agent(client, settings.llm_endpoint, backend)
