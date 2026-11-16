# Prep kit (before the interview)

Thanks for making time for the live case. It's a 60-minute session: you'll walk us through how a small
retrieval-augmented agent works and where it fails, then spend about 20 minutes hands-on in a Databricks workspace
improving it and proving the improvement with evaluations and traces. AI coding assistants are allowed.

## Workspace access
You'll get access to the Databricks workspace at the start of the interview (we'll send an invitation to your
email beforehand: please accept it so you can sign in). Your folder will hold the agent, its tools, the five eval
questions and MLflow traces of the current agent failing them. We'll walk through it together first, then you
improve the agent and prove it with evals.

**Before the interview (optional):** the code is public at https://github.com/dunsky-consulting/jsa-interview-case.
Read as much of it as you like; you can't run it outside the workspace. The five eval questions and the traces of the
agent failing them are shown first in the interview, where we'll ask you to predict which questions fail and why.
`reference.md` and `hello_agent.py` below make the tooling (tool calling, MLflow traces, `mlflow.genai.evaluate`)
familiar. No need to install anything.

## Also in this kit
| File | Purpose |
|---|---|
| `check_access.py` | Confirms you can reach the LLM endpoint, the SQL warehouse and MLflow. We'll run it together at the start of the interview. |
| `hello_agent.py` | A minimal tool-calling agent (OpenAI SDK + MLflow tracing + `mlflow.genai.evaluate`) on toy data, to read if you want to see the tooling. |
| `reference.md` | Document titles in the library, tool signatures and the scoring dimensions. |

Stack: Azure Databricks (Unity Catalog, Vector Search, Model Serving), an open-weight LLM (gpt-oss-120b),
MLflow 3 tracing and `mlflow.genai.evaluate`, plain Python.
