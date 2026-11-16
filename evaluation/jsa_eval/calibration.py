"""Calibration: keep only questions the baseline fails consistently and a reasonable fix passes.

Inputs are ``EvaluationResult.result_df`` frames from repeated runs (one per run).
"""

import json
from collections import defaultdict
from dataclasses import dataclass

from jsa_eval.evalsets import EvalCase
from jsa_eval.scorers import case_passed


def _truthy(value) -> bool | None:
    if value is None or (isinstance(value, float) and value != value):  # NaN from missing assessments
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "true", "pass"}
    return bool(value)


def row_passed(row: dict) -> bool:
    """Pass/fail for one ``result_df`` row (rows that errored have no assessments and fail)."""
    return case_passed(
        not_found_ok=bool(_truthy(row.get("correct_not_found/value"))),
        guidelines_ok=_truthy(row.get("expectations_guidelines/value")),
    )


def _question(request) -> str | None:
    """The question from a row's ``request`` (a dict, or its JSON string)."""
    if isinstance(request, str):
        try:
            request = json.loads(request)
        except json.JSONDecodeError:
            return None
    return request.get("question") if isinstance(request, dict) else None


def _rows_by_case(result_df, cases: list[EvalCase]) -> dict[str, dict]:
    """``result_df`` rows keyed by case id.

    Rows whose agent call errored carry no expectations, so ``case_id/value`` is missing;
    those are matched back to their case by question text.
    """
    ids_by_question = {c.question: c.id for c in cases}
    rows = {}
    for row in result_df.to_dict(orient="records"):
        case_id = row.get("case_id/value")
        if not isinstance(case_id, str):
            case_id = ids_by_question.get(_question(row.get("request")))
        if case_id is not None:
            rows[case_id] = row
    return rows


def case_outcomes(result_df, cases: list[EvalCase]) -> dict[str, bool]:
    """Pass/fail per case id for one eval run (errored rows count as failures)."""
    return {case_id: row_passed(row) for case_id, row in _rows_by_case(result_df, cases).items()}


TABLE_SCORERS = ["correct_not_found", "expectations_guidelines", "retrieval_groundedness", "relevance_to_query",
                 "tool_call_efficiency", "safety", "citation_present"]


def results_table(result_df, cases: list[EvalCase]):
    """One readable row per question: result, answer, the pass/fail judge's reason, every scorer, trace id."""
    import pandas as pd

    rows = _rows_by_case(result_df, cases)
    records = []
    for case in cases:
        row = rows.get(case.id, {})
        records.append({
            "question": case.id,
            "result": "PASS" if row and row_passed(row) else "FAIL",
            "answer": str(row.get("response") or "")[:400],
            "why (expectations_guidelines)": row.get("expectations_guidelines/rationale"),
            **{name: row.get(f"{name}/value") for name in TABLE_SCORERS},
            "trace_id": row.get("trace_id"),
        })
    return pd.DataFrame(records)


def failing_trace_ids(result_df) -> list[str]:
    """Trace ids of the rows that failed, in row order."""
    return [row["trace_id"] for row in result_df.to_dict(orient="records") if not row_passed(row)]


def pass_rates(runs: list[dict[str, bool]]) -> dict[str, float]:
    """Fraction of runs each case passed."""
    totals: dict[str, list[bool]] = defaultdict(list)
    for run in runs:
        for case_id, passed in run.items():
            totals[case_id].append(passed)
    return {case_id: sum(results) / len(results) for case_id, results in totals.items()}


@dataclass(frozen=True)
class CalibrationVerdict:
    case_id: str
    baseline_pass_rate: float
    reference_pass_rate: float
    keep: bool
    reason: str


def select_cases(
    baseline_runs: list[dict[str, bool]],
    reference_runs: list[dict[str, bool]],
    max_baseline_pass_rate: float = 0.2,  # one lucky pass in five runs
    min_reference_pass_rate: float = 0.8,
) -> list[CalibrationVerdict]:
    """Keep a case when the baseline (almost) never passes and the reference agent reliably does."""
    baseline, reference = pass_rates(baseline_runs), pass_rates(reference_runs)
    verdicts = []
    for case_id in sorted(baseline):
        b, r = baseline[case_id], reference.get(case_id, 0.0)
        if b > max_baseline_pass_rate:
            keep, reason = False, "baseline passes sometimes: not a consistent fail"
        elif r < min_reference_pass_rate:
            keep, reason = False, "reference agent fails too: question may be broken or unfair"
        else:
            keep, reason = True, "consistent baseline fail, fixable"
        verdicts.append(CalibrationVerdict(case_id, b, r, keep, reason))
    return verdicts
