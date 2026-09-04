"""Safe calculator tool for the Week 4 agent."""

import ast
import math
import operator

from langchain.tools import tool


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if not math.isfinite(float(node.value)):
            raise ValueError("Only finite numbers are allowed.")
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent is limited to 10.")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    raise ValueError("Unsupported expression. Use numbers and + - * / // % ** with parentheses.")


@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression safely.

    Use this tool for arithmetic rather than doing calculations mentally.
    """
    expression = expression.strip()
    if not expression or len(expression) > 200:
        return "Error: expression must be 1-200 characters."
    try:
        value = _eval(ast.parse(expression, mode="eval"))
        if not math.isfinite(value):
            raise ValueError("Result is not finite.")
        return str(int(value)) if value.is_integer() else f"{value:.10g}"
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"Error: {exc}"
