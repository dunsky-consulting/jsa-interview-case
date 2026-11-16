"""Agent tools: the UC functions in ``tools/*.sql`` and a thin client to call them.

The agent depends on the ``ToolBackend`` protocol only, so tests (and the mock-LLM
local run) swap in an in-memory backend without touching agent code.
"""

from pathlib import Path
from typing import Any, Protocol

from jsa.config import Settings

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
TOOL_FUNCTIONS = ("search_reports", "get_report_metadata")


class ToolBackend(Protocol):
    def search_reports(
        self,
        query: str,
        k: int = 3,
        utility: str | None = None,
        doc_type: str | None = None,
        jurisdiction: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_report_metadata(self, doc_id: str | None = None) -> list[dict[str, Any]]: ...


def render_tool_sql(function: str, settings: Settings) -> str:
    """Fill the brace placeholders of ``tools/<function>.sql`` from settings."""
    sql = (TOOLS_DIR / f"{function}.sql").read_text(encoding="utf-8")
    values = {
        "catalog": settings.catalog,
        "schema": settings.schema,
        "index_name": settings.index_name,
        "embedding_endpoint": settings.embedding_endpoint,
        "documents_table": settings.documents_table,
    }
    for key, value in values.items():
        sql = sql.replace("{" + key + "}", value)
    return sql


def as_retriever_documents(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape search rows as MLflow retriever documents.

    The retrieval hit@k scorer reads ``metadata.doc_id`` from RETRIEVER spans, so any
    agent that wants that score must return this shape from its retriever span.
    """
    return [
        {
            "page_content": row.get("text", ""),
            "metadata": {
                "doc_id": row.get("doc_id"),
                "page": row.get("page"),
                "chunk_id": row.get("chunk_id"),
                "doc_uri": f"{row.get('doc_id')}#page={row.get('page')}",
            },
        }
        for row in rows
    ]


_PARAM_TYPES = {int: "INT", str: "STRING"}


class UCFunctionBackend:
    """Calls the UC functions through the SQL Statement Execution API on a SQL warehouse."""

    def __init__(self, settings: Settings, workspace_client=None):
        if workspace_client is None:
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient()
        self._w = workspace_client
        self._settings = settings

    def _call(self, function: str, **args: Any) -> list[dict[str, Any]]:
        from databricks.sdk.service.sql import StatementParameterListItem, StatementState

        args = {k: v for k, v in args.items() if v is not None}
        named = ", ".join(f"{k} => :{k}" for k in args)
        response = self._w.statement_execution.execute_statement(
            statement=f"SELECT * FROM {self._settings.fqn(function)}({named})",
            warehouse_id=self._settings.warehouse_id,
            parameters=[
                StatementParameterListItem(name=k, value=str(v), type=_PARAM_TYPES[type(v)]) for k, v in args.items()
            ],
            wait_timeout="50s",
        )
        if response.status.state != StatementState.SUCCEEDED:
            error = response.status.error
            raise RuntimeError(f"{function} failed ({response.status.state}): {error.message if error else ''}")
        columns = [c.name for c in response.manifest.schema.columns]
        rows = response.result.data_array if response.result and response.result.data_array else []
        return [dict(zip(columns, row)) for row in rows]

    def search_reports(self, query, k=3, utility=None, doc_type=None, jurisdiction=None):
        rows = self._call(
            "search_reports", query=query, k=k, utility=utility, doc_type=doc_type, jurisdiction=jurisdiction
        )
        for row in rows:
            row["page"] = int(row["page"]) if row.get("page") is not None else None
            row["score"] = float(row["score"]) if row.get("score") is not None else None
        return rows

    def get_report_metadata(self, doc_id=None):
        return self._call("get_report_metadata", doc_id=doc_id)
