"""In-workspace embedding model: multilingual-e5 served from Unity Catalog.

E5 expects ``"query: "`` / ``"passage: "`` prefixes; callers add them (search_reports.sql
adds ``query:``, 00_setup adds ``passage:``) so the endpoint itself stays a plain encoder.
Requires ``sentence-transformers`` (installed by 00_setup only).
"""

from datetime import timedelta
from pathlib import Path

import mlflow


# Logged as a model-from-code file so the serving container never needs the jsa package.
MODEL_CODE = Path(__file__).with_name("e5_model_code.py")


def register_e5(hf_model_id: str, uc_model_name: str) -> str:
    """Download ``hf_model_id``, log it as a pyfunc and register it in UC; returns the new version."""
    import tempfile

    from mlflow.models import infer_signature
    from sentence_transformers import SentenceTransformer

    local_dir = tempfile.mkdtemp(prefix="e5_")  # serverless: only temp dirs are writable
    model = SentenceTransformer(hf_model_id, device="cpu")
    model.save(local_dir)
    example = ["query: hosting capacity map"]
    signature = infer_signature(example, model.encode(example, normalize_embeddings=True))
    mlflow.set_registry_uri("databricks-uc")
    with mlflow.start_run(run_name=f"register {hf_model_id}"):
        info = mlflow.pyfunc.log_model(
            name="e5",
            python_model=str(MODEL_CODE),
            artifacts={"model_dir": local_dir},
            signature=signature,
            input_example=example,
            pip_requirements=["sentence-transformers>=3,<6", "torch", "mlflow>=3.17,<4"],
            registered_model_name=uc_model_name,
        )
    return str(info.registered_model_version)


def latest_registered_version(uc_model_name: str) -> str | None:
    """Highest registered version of the UC model, or None if it was never registered."""
    from mlflow import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    try:
        versions = MlflowClient().search_model_versions(f"name = '{uc_model_name}'")
    except mlflow.exceptions.MlflowException:
        return None
    return str(max(int(v.version) for v in versions)) if versions else None


# A first build (image with torch + 1 GB of weights) can exceed the SDK's 20-minute default.
ENDPOINT_TIMEOUT = timedelta(minutes=45)


def ensure_embedding_endpoint(workspace_client, endpoint_name: str, uc_model_name: str, version: str) -> None:
    """Create the CPU serving endpoint, or point an existing one at ``version``."""
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

    entity = ServedEntityInput(
        entity_name=uc_model_name, entity_version=version, workload_size="Small", scale_to_zero_enabled=True
    )
    existing = {e.name for e in workspace_client.serving_endpoints.list()}
    if endpoint_name in existing:
        workspace_client.serving_endpoints.update_config_and_wait(
            endpoint_name, served_entities=[entity], timeout=ENDPOINT_TIMEOUT
        )
    else:
        workspace_client.serving_endpoints.create_and_wait(
            endpoint_name,
            config=EndpointCoreConfigInput(name=endpoint_name, served_entities=[entity]),
            timeout=ENDPOINT_TIMEOUT,
        )
