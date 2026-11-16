"""Corpus manifest: the single list of documents the index may contain.

Anything not listed here never reaches the index, and anything matching an
``excluded`` pattern aborts the run even if someone lists it by mistake.
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "config" / "corpus_manifest.yml"
# Kept outside config/ because the excluded file names are themselves confidential.
EXCLUDED_PATH = PROJECT_ROOT / "interviewer" / "excluded_sources.yml"

DOC_TYPES = {"application", "decision", "plan", "guideline", "research", "order"}
ACCESS_LEVELS = {"public", "internal"}


class ExcludedDocumentError(ValueError):
    """Raised when a confidential/excluded file is about to be ingested."""


@dataclass(frozen=True)
class Document:
    """Index-level metadata for one document (see CLAUDE.md, metadata schema)."""

    doc_id: str
    title: str
    publisher: str
    utility: str | None
    jurisdiction: str
    doc_type: str
    proceeding: str | None
    date: str | None
    language: str
    source_path: str
    superseded_by: str | None = None
    access: str = "public"

    def as_row(self) -> dict:
        """Plain dict for building a Spark/Delta row."""
        return asdict(self)


METADATA_COLUMNS = [f.name for f in fields(Document)]


@dataclass(frozen=True)
class Manifest:
    documents: list[Document]
    excluded: list[str]

    def by_id(self) -> dict[str, Document]:
        return {d.doc_id: d for d in self.documents}


def check_not_excluded(source_path: str, excluded: list[str]) -> None:
    """Raise if ``source_path`` matches any excluded pattern (case-insensitive substring)."""
    normalized = source_path.replace("\\", "/").lower()
    for pattern in excluded:
        if pattern.lower() in normalized:
            raise ExcludedDocumentError(f"{source_path!r} matches excluded pattern {pattern!r}")


def validate(manifest: Manifest) -> None:
    """Fail fast on excluded paths, duplicate ids, bad enums and dangling superseded_by."""
    ids = [d.doc_id for d in manifest.documents]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate doc_id(s): {sorted(duplicates)}")
    known = set(ids)
    for doc in manifest.documents:
        check_not_excluded(doc.source_path, manifest.excluded)
        if doc.doc_type not in DOC_TYPES:
            raise ValueError(f"{doc.doc_id}: doc_type {doc.doc_type!r} not in {sorted(DOC_TYPES)}")
        if doc.access not in ACCESS_LEVELS:
            raise ValueError(f"{doc.doc_id}: access {doc.access!r} not in {sorted(ACCESS_LEVELS)}")
        if doc.superseded_by and doc.superseded_by not in known:
            raise ValueError(f"{doc.doc_id}: superseded_by {doc.superseded_by!r} is not in the manifest")


def load_manifest(path: Path | str = MANIFEST_PATH, excluded_path: Path | str = EXCLUDED_PATH) -> Manifest:
    """Load and validate the YAML manifest, guarded by the exclusion list when it is deployed."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    excluded_file = Path(excluded_path)
    excluded = yaml.safe_load(excluded_file.read_text(encoding="utf-8"))["excluded"] if excluded_file.is_file() else []
    manifest = Manifest(
        documents=[Document(**entry) for entry in raw["documents"]],
        excluded=list(excluded),
    )
    validate(manifest)
    return manifest
