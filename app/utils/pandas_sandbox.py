from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

# Coarse denylist checked against the raw source text before it's even
# parsed - defense in depth on top of the restricted builtins/AST check below.
_BLOCKED_PATTERNS = [
    r"\bimport\b", r"\b__\w+__\b", r"\bexec\b", r"\beval\b", r"\bopen\b",
    r"\bos\.", r"\bsys\.", r"\bsubprocess\b", r"\bshutil\b", r"\bsocket\b",
    r"\bglobals\b", r"\blocals\b", r"\bgetattr\b", r"\bsetattr\b", r"\bdelattr\b",
    r"\bcompile\b", r"\binput\b", r"\bbreakpoint\b", r"\bexit\b", r"\bquit\b",
    r"\bpip\b", r"\brequests\b", r"\bstreamlit\b", r"\bst\.",
]
_BLOCKED_RE = re.compile("|".join(_BLOCKED_PATTERNS))

# Small, explicit allowlist of builtins - enough for typical pandas one-liners
# (len, range, sorted, etc.) without exposing file/process/import primitives.
_ALLOWED_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
    "list", "max", "min", "range", "reversed", "round", "set", "sorted",
    "str", "sum", "tuple", "zip", "True", "False", "None",
)
_SAFE_BUILTINS = {name: getattr(builtins, name) for name in _ALLOWED_BUILTIN_NAMES if hasattr(builtins, name)}


@dataclass
class SnippetResult:
    """Outcome of running a user snippet: exactly one of `value`/`error` is set."""

    value: Optional[Any] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_pandas_snippet(code: str, df: pd.DataFrame) -> SnippetResult:
    """Execute a short pandas snippet against `df` and return its result.

    The snippet may be a single expression (`df[df['age'] > 30]`) or a few
    statements ending in an expression / a `result = ...` assignment - like a
    Jupyter cell, the value of the trailing expression (if any) is returned,
    otherwise the `result` variable is looked up.
    """
    code = code.strip()
    if not code:
        return SnippetResult(error=f"Write a pandas expression first, e.g. df[df['age'] > 30], Code : {code}")

    if _BLOCKED_RE.search(code):
        return SnippetResult(error="That snippet uses a disallowed keyword (imports, dunders, and I/O aren't permitted here).")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return SnippetResult(error=f"Syntax error: {exc}")

    trailing_expr = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        trailing_expr = ast.Expression(tree.body.pop().value)

    namespace = {"df": df, "pd": pd, "np": np}
    try:
        if tree.body:
            exec(compile(tree, "<pandas_filter>", "exec"), {"__builtins__": _SAFE_BUILTINS}, namespace)
        if trailing_expr is not None:
            value = eval(compile(trailing_expr, "<pandas_filter_expr>", "eval"), {"__builtins__": _SAFE_BUILTINS}, namespace)
        else:
            value = namespace.get("result")
            if value is None:
                return SnippetResult(error="No trailing expression and no `result = ...` variable was set.")
    except Exception as exc:  # noqa: BLE001 - surface any runtime error from the snippet itself
        return SnippetResult(error=f"{type(exc).__name__}: {exc}")

    return SnippetResult(value=value)