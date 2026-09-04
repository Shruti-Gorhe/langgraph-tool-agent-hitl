Tool-Using Agent + HITL

This project extends the Week 3 RAG benchmark instead of replacing it.

## What was added

### 1. Tool-using agent

The agent has four tools (three required categories plus a protected write tool):

1. `enterprise_search` — reuses the Week 3 Chroma/BM25 retrieval stack.
2. `calculator` — safe arithmetic evaluation.
3. `get_employee_info` — read-only mock internal HR service.
4. `update_employee_record` — write operation used to demonstrate HITL.

The agent is built with LangChain's current `create_agent` API and uses LangGraph
checkpointing for approval/resume behavior.

### 2. Human-in-the-loop

`update_employee_record` is protected by `HumanInTheLoopMiddleware`. The tool call is
paused before execution and the reviewer can approve or reject it.

The Streamlit reviewer UI displays the requested tool and arguments and provides
**Approve** / **Reject** controls.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

This demo uses Ollama by default so no cloud API key is required. Start Ollama and
install a tool-calling model, then optionally set `OLLAMA_MODEL`.

```bash
ollama pull llama3.2
```

## Step 1 — Build the Week 3 retrieval index

Run the original benchmark pipeline first:

```bash
python run.py
```

This populates the existing Chroma collections under `storage/chroma`.

## Step 2 — Run the CLI agent

```bash
python scripts/run_agent.py "What is the annual home-office stipend for remote workers?"
```

Calculator example:

```bash
python scripts/run_agent.py "If the home-office stipend is 500 dollars per year, what is the monthly equivalent?"
```

Internal API example:

```bash
python scripts/run_agent.py "What is EMP001's leave balance?"
```

HITL example:

```bash
python scripts/run_agent.py "Update EMP001's leave balance to 20 days."
```

For the CLI demo, add `--approve` to approve the pending write or `--reject` to reject it.

## Step 3 — Run the reviewer UI

```bash
streamlit run streamlit_app.py
```

Try:

> Update EMP001's leave balance to 20 days.

The agent should stop at the approval gate. Review the proposed arguments and click
**Approve** or **Reject**.

## Observable trace

The UI/CLI shows observable execution events:

```text
User Query
   ↓
Tool Call + Arguments
   ↓
Tool Result
   ↓
Final Answer
```

This deliberately avoids exposing hidden chain-of-thought. The deliverable only needs
traceable tool/action execution.

## Environment variables

```text
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
AGENT_RAG_STRATEGY=hierarchical
AGENT_RAG_METHOD=hybrid
```

The RAG strategy/method can be changed after reviewing the Week 3 benchmark results.

##SAMPLE COMPLEX QUERY

For EMP001, retrieve their current leave balance and check the Northwind leave policy to determine whether they can take 7 working days of leave. Calculate the percentage of their current balance that would remain afterward. If they would still have at least 50% of their original balance remaining and there is no policy restriction preventing it, calculate the new leave balance and prepare an update to EMP001’s record. Do not execute the update without human approval. In your final response, clearly distinguish between what you verified, what you calculated, and what is waiting for approval.
