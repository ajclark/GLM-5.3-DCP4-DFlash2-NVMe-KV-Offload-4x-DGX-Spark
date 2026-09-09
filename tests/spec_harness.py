"""CPU tests load real source without importing vLLM/CUDA or replacing packages."""
import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/spec_runtime"


def load_policy():
    name = "glm_adaptive_policy_test"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "overlay/vllm/v1/spec_decode/adaptive.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def source_class(path, name, methods, namespace, bases=None):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    assert {n.name for n in node.body} == set(methods)
    if bases is not None:
        node.bases = [ast.Name(id=b, ctx=ast.Load()) for b in bases]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]
