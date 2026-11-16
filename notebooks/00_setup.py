# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup: parse, chunk, embed, index
# MAGIC
# MAGIC Builds the report library from the files uploaded to the raw volume
# MAGIC (`scripts/upload_corpus.py`). Re-running rebuilds everything from scratch.
# MAGIC
# MAGIC 1. Schema + raw volume
# MAGIC 2. `documents` table from `config/corpus_manifest.yml`
# MAGIC 3. `pages` table (one row per PDF page)
# MAGIC 4. `chunks` table: fixed-size chunks + metadata + e5 embeddings
# MAGIC 5. AI Search (Vector Search) endpoint + Delta Sync index with self-managed embeddings

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt "sentence-transformers>=3,<6"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import time
from datetime import timedelta

sys.path.insert(0, os.path.abspath("../src"))

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    EndpointType,
    PipelineType,
    VectorIndexType,
)

from jsa.chunking import naive_chunks
from jsa.config import settings_from_widgets
from jsa.corpus import METADATA_COLUMNS, check_not_excluded, load_manifest
from jsa.embedding_model import ensure_embedding_endpoint, latest_registered_version, register_e5
from jsa.parsing import extract_document_pages

settings = settings_from_widgets(dbutils)
dbutils.widgets.text("embedding_model", "intfloat/multilingual-e5-base")
dbutils.widgets.dropdown("reregister_model", "false", ["true", "false"])
EMBEDDING_MODEL = dbutils.widgets.get("embedding_model")
REREGISTER_MODEL = dbutils.widgets.get("reregister_model") == "true"  # otherwise reuse the latest UC version
w = WorkspaceClient()
mlflow.set_experiment(settings.experiment_path)  # model registration logs a run

# COMMAND ----------

# MAGIC %md ## 1. Schema and raw volume

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {settings.catalog}.{settings.schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {settings.fqn('raw_reports')}")

manifest = load_manifest()
missing = [d.source_path for d in manifest.documents if not os.path.exists(f"{settings.raw_volume_path}/{d.source_path}")]
if missing:
    raise FileNotFoundError(f"Upload these to {settings.raw_volume_path} first (scripts/upload_corpus.py): {missing}")

# COMMAND ----------

# MAGIC %md ## 2–3. Documents and pages

# COMMAND ----------

documents_df = spark.createDataFrame([d.as_row() for d in manifest.documents], schema=", ".join(f"{c} string" for c in METADATA_COLUMNS))
documents_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(settings.documents_table)

page_rows = []
for doc in manifest.documents:
    check_not_excluded(doc.source_path, manifest.excluded)
    with open(f"{settings.raw_volume_path}/{doc.source_path}", "rb") as f:
        pages = extract_document_pages(doc.source_path, f.read())
    page_rows += [(doc.doc_id, number, text) for number, text in pages]
    print(f"{doc.doc_id}: {len(pages)} pages")

spark.createDataFrame(page_rows, "doc_id string, page int, text string").write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(settings.pages_table)

# COMMAND ----------

# MAGIC %md ## 4. Chunks + embeddings
# MAGIC Embeddings come from the in-workspace multilingual-e5 endpoint (passages prefixed with `passage: `).

# COMMAND ----------

uc_model = settings.fqn("multilingual_e5")
version = None if REREGISTER_MODEL else latest_registered_version(uc_model)
if version is None:
    version = register_e5(EMBEDDING_MODEL, uc_model)
print(f"serving {uc_model} version {version}")
ensure_embedding_endpoint(w, settings.embedding_endpoint, uc_model, version)

pages_by_doc: dict[str, list[tuple[int, str]]] = {}
for doc_id, number, text in page_rows:
    pages_by_doc.setdefault(doc_id, []).append((number, text))
chunk_rows = [(c.chunk_id, c.doc_id, c.page, c.text) for doc_id, pages in pages_by_doc.items() for c in naive_chunks(doc_id, pages)]
spark.createDataFrame(chunk_rows, "chunk_id string, doc_id string, page int, text string").createOrReplaceTempView("chunks_staging")
print(f"{len(chunk_rows)} chunks")

# The index must be dropped before its source table is replaced.
endpoints = {e.name for e in w.vector_search_endpoints.list_endpoints()}
if settings.vector_search_endpoint in endpoints:
    indexes = {i.name for i in w.vector_search_indexes.list_indexes(settings.vector_search_endpoint)}
    if settings.index_name in indexes:
        w.vector_search_indexes.delete_index(settings.index_name)

metadata_select = ", ".join(f"d.{c}" for c in METADATA_COLUMNS if c != "doc_id")
spark.sql(f"""
CREATE OR REPLACE TABLE {settings.chunks_table}
TBLPROPERTIES (delta.enableChangeDataFeed = true)
AS SELECT c.chunk_id, c.doc_id, c.page, c.text, {metadata_select},
          ai_query('{settings.embedding_endpoint}', concat('passage: ', c.text), returnType => 'ARRAY<FLOAT>') AS embedding
FROM chunks_staging c JOIN {settings.documents_table} d USING (doc_id)
""")
dimension = len(spark.table(settings.chunks_table).select("embedding").first()["embedding"])
print(f"embedding dimension {dimension}")

# COMMAND ----------

# MAGIC %md ## 5. Vector Search endpoint and index

# COMMAND ----------

if settings.vector_search_endpoint not in endpoints:
    w.vector_search_endpoints.create_endpoint_and_wait(
        settings.vector_search_endpoint, endpoint_type=EndpointType.STANDARD, timeout=timedelta(minutes=45)
    )

w.vector_search_indexes.create_index(
    name=settings.index_name,
    endpoint_name=settings.vector_search_endpoint,
    primary_key="chunk_id",
    index_type=VectorIndexType.DELTA_SYNC,
    delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
        source_table=settings.chunks_table,
        pipeline_type=PipelineType.TRIGGERED,
        embedding_vector_columns=[EmbeddingVectorColumn(name="embedding", embedding_dimension=dimension)],
        columns_to_sync=["chunk_id", "doc_id", "page", "text", *[c for c in METADATA_COLUMNS if c != "doc_id"], "embedding"],
    ),
)
deadline = time.time() + 3600
while not (status := w.vector_search_indexes.get_index(settings.index_name).status).ready:
    if time.time() > deadline:
        raise TimeoutError(f"index not ready after 1h: {status.message}")
    time.sleep(30)
print(f"index ready: {status.indexed_row_count} rows")
