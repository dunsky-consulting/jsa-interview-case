"""Eval set files (JSONL, one case per line) and conversion to MLflow eval records."""

import json
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISIBLE_SET = PROJECT_ROOT / "eval" / "visible.jsonl"


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    category: str
    answerable: bool
    guidelines: list[str]  # plain-English rules a good answer meets; the guidelines judge checks every one
    expected_doc_ids: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    reference_answer: str = ""
    notes: str = ""

    def to_record(self) -> dict:
        """MLflow ``genai.evaluate`` row: inputs feed predict_fn(question=...), expectations feed scorers."""
        return {
            "inputs": {"question": self.question},
            "expectations": {
                "case_id": self.id,
                "guidelines": self.guidelines,
                "expected_doc_ids": self.expected_doc_ids,
                "answerable": self.answerable,
            },
        }


def load_eval_set(path: Path | str = VISIBLE_SET) -> list[EvalCase]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [EvalCase(**json.loads(line)) for line in lines if line.strip()]
