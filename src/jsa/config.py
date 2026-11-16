"""Runtime settings. Every workspace-specific name comes from here, never from code.

In jobs/notebooks the values arrive as job parameters (widgets) set from bundle
variables; locally they come from ``JSA_*`` environment variables.
"""

import os
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    catalog: str
    schema: str
    llm_endpoint: str
    embedding_endpoint: str
    vector_search_endpoint: str
    warehouse_id: str
    experiment_path: str
    judge_endpoint: str = ""

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> "Settings":
        """Build from any mapping (widgets dict, env subset); unknown keys are ignored."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names and v is not None})

    @classmethod
    def from_env(cls, prefix: str = "JSA_") -> "Settings":
        """Build from ``JSA_CATALOG``, ``JSA_SCHEMA``, ... environment variables."""
        return cls.from_mapping(
            {f.name: os.environ[prefix + f.name.upper()] for f in fields(cls) if prefix + f.name.upper() in os.environ}
        )

    def fqn(self, name: str) -> str:
        """Fully-qualified ``catalog.schema.name``."""
        return f"{self.catalog}.{self.schema}.{name}"

    @property
    def judge_model(self) -> str:
        """MLflow judge URI; defaults to the workspace LLM endpoint instead of Databricks-managed judges."""
        return f"databricks:/{self.judge_endpoint or self.llm_endpoint}"

    @property
    def raw_volume_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/raw_reports"

    @property
    def documents_table(self) -> str:
        return self.fqn("documents")

    @property
    def pages_table(self) -> str:
        return self.fqn("pages")

    @property
    def chunks_table(self) -> str:
        return self.fqn("chunks")

    @property
    def index_name(self) -> str:
        return self.fqn("chunks_index")


WORKSPACE_DEFAULTS = Path(__file__).resolve().parents[2] / "config" / "workspace.yml"


def workspace_defaults(path: Path = WORKSPACE_DEFAULTS) -> dict[str, str]:
    """Defaults for interactive notebook runs (config/workspace.yml).

    A relative ``experiment_path`` is resolved against the deployed bundle files folder (the parent of ``config/``),
    so each candidate deployment defaults to its own ``<root_path>/experiment``.
    """
    values = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    defaults = {k: "" if v is None else str(v) for k, v in values.items()}
    experiment = defaults.get("experiment_path", "")
    if experiment and not experiment.startswith("/"):
        files_root = path.resolve().parents[1].as_posix()
        defaults["experiment_path"] = posixpath.normpath(posixpath.join(files_root, experiment))
    return defaults


def settings_from_widgets(dbutils) -> Settings:
    """One text widget per setting, pre-filled from config/workspace.yml; job parameters override them."""
    defaults = workspace_defaults()
    for f in fields(Settings):
        dbutils.widgets.text(f.name, defaults.get(f.name, ""))
    return Settings.from_mapping({f.name: dbutils.widgets.get(f.name) for f in fields(Settings)})
