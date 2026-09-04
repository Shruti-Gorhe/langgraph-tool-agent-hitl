"""CLI demo for the Week 4 tool-using agent."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.types import Command

from src.agent.tool_agent import get_agent
from src.agent.trace import extract_trace, final_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week 4 enterprise agent")
    parser.add_argument("query", nargs="+", help="User query")
    parser.add_argument("--approve", action="store_true", help="Approve a pending write")
    parser.add_argument("--reject", action="store_true", help="Reject a pending write")
    args = parser.parse_args()

    query = " ".join(args.query)
    agent = get_agent()
    config = {"configurable": {"thread_id": "cli-demo"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
        version="v2",
    )

    if getattr(result, "interrupts", None):
        print("\n=== HUMAN APPROVAL REQUIRED ===")
        for interrupt in result.interrupts:
            print(json.dumps(interrupt.value, indent=2, default=str))
        if args.approve or args.reject:
            decision = "approve" if args.approve else "reject"
            result = agent.invoke(
                Command(resume={"decisions": [{"type": decision}]}),
                config=config,
                version="v2",
            )
        else:
            print("\nRe-run with --approve or --reject using the same thread in a persistent UI.")
            return

    print("\n=== TRACE ===")
    for item in extract_trace(result):
        print(json.dumps(item, indent=2, default=str))

    print("\n=== FINAL ANSWER ===")
    print(final_text(result))


if __name__ == "__main__":
    main()
