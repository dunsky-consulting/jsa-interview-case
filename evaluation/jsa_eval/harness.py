"""Eval harness: runs any agent module that follows the ``build_agent`` contract.

Contract: ``module.build_agent(settings, client=None, backend=None)`` returns an object
with ``answer(question) -> str``. The harness injects the LLM client so it can cap LLM
calls per question; the cap protects endpoint capacity and is not part of the agent.
"""

import importlib
import os
import sys
from types import SimpleNamespace

import mlflow

from jsa.config import Settings
from jsa_eval.evalsets import EvalCase
from jsa_eval.scorers import build_scorers

DEFAULT_MAX_LLM_CALLS = 25


class CallBudgetExceeded(RuntimeError):
    """The agent made more LLM calls than the harness allows for one question."""


class BudgetedClient:
    """OpenAI-compatible client proxy that raises after ``max_calls`` chat completions."""

    def __init__(self, client, max_calls: int):
        self._client = client
        self._max_calls = max_calls
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        if self.calls > self._max_calls:
            raise CallBudgetExceeded(f"harness stopped the agent after {self._max_calls} LLM calls")
        return self._client.chat.completions.create(**kwargs)


def import_fresh(module_name: str):
    """Import ``module_name`` after dropping its cached top-level package, so edits made since the
    last run are picked up in a long-lived notebook session. The eval package itself is kept."""
    package = module_name.split(".")[0]
    for name in [n for n in sys.modules if n == package or n.startswith(package + ".")]:
        del sys.modules[name]
    importlib.invalidate_caches()
    return importlib.import_module(module_name)


def make_predict_fn(settings: Settings, agent_module: str, client, backend, max_llm_calls: int):
    """predict_fn for ``mlflow.genai.evaluate``; a fresh agent and budget per question (thread-safe)."""
    module = import_fresh(agent_module)

    def predict_fn(question: str) -> str:
        agent = module.build_agent(settings, client=BudgetedClient(client, max_llm_calls), backend=backend)
        return agent.answer(question)

    return predict_fn


# MLflow imports these lazily from scorer, judge and span-logging code. Under many evaluate()
# worker threads those first imports (and MLflow's import hooks) deadlock on Python's import
# locks; importing them once up front, in the calling thread, avoids it.
MLFLOW_LAZY_IMPORTS = (
    "mlflow.genai.scorers.job",
    "mlflow.server.jobs.utils",
    "mlflow.genai.judges.utils.telemetry_utils",
    "mlflow.genai.judges.adapters.rate_limit_retry_adapters",
    "mlflow.genai.judges.adapters.litellm_adapter",
    "mlflow.utils.providers",
)


def preload_for_parallel_eval() -> None:
    """Disable MLflow usage telemetry and pre-import MLflow's lazily imported modules."""
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    # evaluate() otherwise runs the first question alone before going parallel, adding a full question's latency.
    os.environ.setdefault("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "true")
    for module in MLFLOW_LAZY_IMPORTS:
        importlib.import_module(module)


# Parallel eval bursts can hit the provider's per-minute limits; a 429 is infrastructure noise,
# not agent quality, so the harness client retries it (exponential backoff, honours Retry-After).
LLM_MAX_RETRIES = 8


def default_clients(settings: Settings):
    """Workspace LLM client (with rate-limit retries) and UC-function tool backend."""
    from databricks.sdk import WorkspaceClient

    from jsa.tools import UCFunctionBackend

    w = WorkspaceClient()
    client = w.serving_endpoints.get_open_ai_client().with_options(max_retries=LLM_MAX_RETRIES)
    return client, UCFunctionBackend(settings, w)


def run_eval(
    cases: list[EvalCase],
    settings: Settings,
    agent_module: str = "jsa.agent",
    include_judge: bool = True,
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
    client=None,
    backend=None,
    run_name: str | None = None,
):
    """Run one evaluation pass in the active experiment and return the MLflow EvaluationResult."""
    if client is None or backend is None:
        default_client, default_backend = default_clients(settings)
        client, backend = client or default_client, backend or default_backend
    predict_fn = make_predict_fn(settings, agent_module, client, backend, max_llm_calls)
    # Run every question at once: wall time is then the slowest question, not a queue.
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", str(max(len(cases), 1)))
    preload_for_parallel_eval()
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"agent_module": agent_module, "n_cases": len(cases)})
        return mlflow.genai.evaluate(
            data=[c.to_record() for c in cases],
            predict_fn=predict_fn,
            scorers=build_scorers(settings, include_judge=include_judge),
        )
