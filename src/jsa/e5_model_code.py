"""MLflow "model from code" for the e5 embedding endpoint.

Logged by file path (see embedding_model.register_e5), so the serving container runs this
file directly. It must stay self-contained: no imports from the jsa package.
"""

import mlflow


class E5Embedder(mlflow.pyfunc.PythonModel):
    """Encodes a column of strings into L2-normalised vectors."""

    def load_context(self, context):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(context.artifacts["model_dir"], device="cpu")

    def predict(self, context, model_input, params=None):
        texts = model_input.iloc[:, 0].tolist() if hasattr(model_input, "iloc") else list(model_input)
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32)


mlflow.models.set_model(E5Embedder())
