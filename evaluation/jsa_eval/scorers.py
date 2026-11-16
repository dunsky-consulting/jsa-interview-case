"""Eval scorers.

Deterministic: retrieval hit@k, correct not-found, citation present, latency, tokens, steps.
LLM judges, one per MLflow category, on ``Settings.judge_model``: expectations_guidelines (per-question rules, decides
pass/fail), retrieval groundedness, relevance to query, tool-call efficiency, safety.

The pure helpers are separate from the MLflow wrappers so they unit-test without a trace store.
"""

import re

from mlflow.entities import Feedback, SpanType
from mlflow.genai.scorers import (
    ExpectationsGuidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    Safety,
    ToolCallEfficiency,
    list_scorers,
    scorer,
)

from jsa.config import Settings
from jsa.corpus import load_manifest

CITATION_PATTERN = re.compile(r"\[\s*([a-z0-9_]+)\s*,\s*p(?:age|p?\.)?\s*(\d+)", re.IGNORECASE)

NOT_FOUND_PATTERNS = [
    r"\bnot found\b",
    r"\bnot (?:in|available in|covered (?:by|in)) (?:the|our|this) (?:library|corpus|documents?|collection)\b",
    r"\b(?:could not|couldn't|cannot|can't|was unable to|am unable to|unable to) (?:find|locate|identify)\b",
    r"\bno (?:relevant |matching )?(?:information|documents?|results?|sources?|evidence)\b",
    r"\b(?:the )?(?:library|corpus|documents?) (?:does|do) not (?:contain|include|cover|mention)\b",
    r"\bcannot (?:share|disclose|provide)\b",
    r"\bintrouvable\b",
    r"\baucun(?:e)? (?:information|document|résultat)\b",
    r"\b(?:ne|n')\s?(?:contient|figure|trouve) pas\b",
]
_NOT_FOUND = re.compile("|".join(NOT_FOUND_PATTERNS), re.IGNORECASE)

LLM_SPAN_TYPES = {SpanType.CHAT_MODEL, SpanType.LLM}


# ---------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------


def retrieved_doc_ids(trace) -> list[str]:
    """Distinct doc_ids from every RETRIEVER span, in retrieval order."""
    seen: list[str] = []
    for span in trace.search_spans(span_type=SpanType.RETRIEVER):
        for document in span.outputs or []:
            doc_id = (document.get("metadata") or {}).get("doc_id") if isinstance(document, dict) else None
            if doc_id and doc_id not in seen:
                seen.append(doc_id)
    return seen


def hit(retrieved: list[str], expected: list[str]) -> bool:
    """True when any expected document was retrieved at least once."""
    return bool(set(retrieved) & set(expected))


def says_not_found(answer: str) -> bool:
    """Heuristic: the answer declines / reports the information is not in the library."""
    return bool(_NOT_FOUND.search(answer or ""))


def citations(answer: str, known_doc_ids: set[str]) -> list[tuple[str, int]]:
    """``[doc_id, p. N]`` citations whose doc_id exists in the library."""
    return [
        (doc_id, int(page))
        for doc_id, page in CITATION_PATTERN.findall(answer or "")
        if doc_id in known_doc_ids
    ]


def llm_call_count(trace) -> int:
    return sum(1 for span in trace.data.spans if span.span_type in LLM_SPAN_TYPES)


def total_tokens(trace) -> int | None:
    usage = trace.info.token_usage or {}
    return usage.get("total_tokens")


def case_passed(not_found_ok: bool, guidelines_ok: bool | None) -> bool:
    """Pass rule used by calibration and summaries: right decline behaviour and the question's guidelines met."""
    return bool(not_found_ok and guidelines_ok)


# ---------------------------------------------------------------------------------------
# MLflow scorers
# ---------------------------------------------------------------------------------------


# Scorers skip a row by returning None. Never return Feedback(value=None): on Databricks it drops every
# assessment on that trace, expectations included.
@scorer
def retrieval_hit(trace, expectations) -> Feedback | None:
    expected = expectations.get("expected_doc_ids") or []
    if not expected:
        return None  # unanswerable question: no expected documents
    retrieved = retrieved_doc_ids(trace)
    return Feedback(value=hit(retrieved, expected), rationale=f"retrieved={retrieved} expected={expected}")


@scorer
def correct_not_found(outputs, expectations) -> Feedback:
    answerable = expectations.get("answerable", True)
    declined = says_not_found(outputs if isinstance(outputs, str) else str(outputs))
    ok = declined != answerable
    expected = "answer" if answerable else "decline / not found"
    return Feedback(value=ok, rationale=f"expected {expected}; agent {'declined' if declined else 'answered'}")


def make_citation_present(known_doc_ids: set[str]):
    @scorer(name="citation_present")
    def citation_present(outputs, expectations) -> Feedback | None:
        if not expectations.get("answerable", True):
            return None  # unanswerable question: citations not required
        found = citations(outputs if isinstance(outputs, str) else str(outputs), known_doc_ids)
        return Feedback(value=bool(found), rationale=f"citations={found}")

    return citation_present


@scorer
def latency_seconds(trace) -> float:
    return round((trace.info.execution_duration or 0) / 1000, 2)


@scorer
def tokens(trace) -> int:
    return total_tokens(trace) or 0


@scorer
def steps(trace) -> int:
    return llm_call_count(trace)


def build_judges(settings: Settings) -> list:
    """One LLM judge per MLflow built-in category, all on the judge endpoint."""
    model = settings.judge_model
    return [
        ExpectationsGuidelines(model=model),  # named expectations_guidelines; `guidelines` is the expectation
        RetrievalGroundedness(model=model),
        RelevanceToQuery(model=model),
        ToolCallEfficiency(model=model),
        Safety(model=model),
    ]


def build_scorers(settings: Settings, include_judge: bool = True) -> list:
    """All scorers for 03_eval / 04_calibration; the judges use the judge endpoint."""
    known = {d.doc_id for d in load_manifest().documents}
    deterministic = [retrieval_hit, correct_not_found, make_citation_present(known), latency_seconds, tokens, steps]
    return deterministic + build_judges(settings) if include_judge else deterministic


def register_judges(settings: Settings, experiment_id: str) -> list[str]:
    """List the judges on the experiment's Scorers page without starting them (no automatic scoring).

    Skips judges already registered, so it is safe to re-run. Returns the names newly registered.
    """
    existing = {s.name for s in list_scorers(experiment_id=experiment_id)}
    added = []
    for judge in build_judges(settings):
        if judge.name not in existing:
            judge.register(name=judge.name, experiment_id=experiment_id)
            added.append(judge.name)
    return added
