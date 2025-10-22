from __future__ import annotations
import ast
from types import ModuleType
import builtins as _bi
from typing import Dict, Any


ALLOWED_IMPORTS = {"voxelsim", "math", "typing"}
ALLOWED_BUILTINS = {
    "abs", "min", "max", "range", "len", "sum", "enumerate", "zip", "map", "filter",
    "all", "any", "isinstance", "print", "float", "int", "bool", "list", "dict", "set", "tuple",
    "__import__",  # needed for import statements executed in module scope
}
DISALLOWED_ATTRS = {"__import__", "eval", "exec", "compile", "open"}


class SandboxError(Exception):
    pass


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".")[0] for alias in node.names]
            for name in modules:
                if name not in ALLOWED_IMPORTS:
                    raise SandboxError(f"Import not allowed: {name}")
        elif isinstance(node, ast.ImportFrom):
            base = (node.module or "").split(".")[0]
            if base and base not in ALLOWED_IMPORTS:
                raise SandboxError(f"Import not allowed: {base}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_ATTRS:
                raise SandboxError(f"Call not allowed: {node.func.id}")
        if isinstance(node, ast.Attribute):
            if node.attr in DISALLOWED_ATTRS:
                raise SandboxError(f"Attribute not allowed: {node.attr}")


def load_policy_module(code: str, module_name: str = "policy") -> ModuleType:
    tree = ast.parse(code, filename=module_name)
    _validate_ast(tree)

    mod = ModuleType(module_name)
    # Restrict builtins
    safe_builtins = {k: getattr(_bi, k) for k in ALLOWED_BUILTINS if hasattr(_bi, k)}
    mod.__dict__["__builtins__"] = safe_builtins
    compiled = compile(tree, filename=module_name, mode="exec")
    exec(compiled, mod.__dict__, mod.__dict__)
    # Verify required symbols
    for fn in ("act", "collect", "finalize"):
        if not callable(mod.__dict__.get(fn)):
            raise SandboxError(f"Required function missing: {fn}")
    return mod
