"""Mock internal HR service used to demonstrate read/write tool calls."""

import json
import threading
from pathlib import Path

from langchain.tools import tool


DATA_FILE = Path("data/employees.json")
_LOCK = threading.Lock()


def _load() -> dict:
    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = DATA_FILE.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    temp.replace(DATA_FILE)


@tool
def get_employee_info(employee_id: str) -> str:
    """Read a fictional employee record from the internal HR service.

    Use this for employee profile and leave-balance lookups. This operation is read-only.
    """
    employee_id = employee_id.strip().upper()
    with _LOCK:
        data = _load()
    employee = data.get(employee_id)
    if employee is None:
        return json.dumps({"error": f"Employee {employee_id} not found."})
    return json.dumps({"employee_id": employee_id, **employee}, indent=2)


@tool
def update_employee_record(employee_id: str, field: str, value: int) -> str:
    """Update a fictional employee record in the internal HR service.

    Use this only when the user explicitly requests a data change.

    IMPORTANT:
    - For leave_balance, value must be a whole number of days.
    - This is a DATA-WRITE operation.
    - Human approval is required before this tool executes.
    """
    employee_id = employee_id.strip().upper()
    field = field.strip()

    if field not in {"leave_balance", "department", "job_title"}:
        return "Error: field is not writable in this demo."

    if field == "leave_balance":
        if not isinstance(value, int):
            return "Error: leave_balance must be an integer number of days."
        if value < 0:
            return "Error: leave_balance cannot be negative."

    with _LOCK:
        data = _load()

        if employee_id not in data:
            return f"Error: employee {employee_id} not found."

        data[employee_id][field] = value
        _save(data)

    return json.dumps(
        {
            "status": "updated",
            "employee_id": employee_id,
            "field": field,
            "value": data[employee_id][field],
        },
        indent=2,
    )
