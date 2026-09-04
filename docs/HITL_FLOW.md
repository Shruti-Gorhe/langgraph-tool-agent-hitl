# Week 4 — Human-in-the-Loop Flow

```mermaid
flowchart TD
    U[User query] --> A[LangChain create_agent]
    A --> D{Select tool}
    D --> RAG[enterprise_search<br/>read-only]
    D --> CALC[calculator<br/>read-only]
    D --> HRR[get_employee_info<br/>read-only]
    D --> WRITE[update_employee_record<br/>DATA WRITE]
    RAG --> A
    CALC --> A
    HRR --> A
    WRITE --> G{HITL approval gate}
    G -->|Approve| EXEC[Execute write tool]
    G -->|Reject| STOP[Return rejection feedback]
    EXEC --> A
    STOP --> A
    A --> F[Final response]
```

## Approval policy

| Tool | Risk | Approval |
|---|---|---|
| `enterprise_search` | Read-only retrieval | No |
| `calculator` | Read-only computation | No |
| `get_employee_info` | Read-only internal lookup | No |
| `update_employee_record` | Data write | **Yes** |

The approval middleware pauses the graph before the write tool executes. The graph is
checkpointed so it can resume with an `approve` or `reject` decision.
