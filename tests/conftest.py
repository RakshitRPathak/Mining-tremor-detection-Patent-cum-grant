"""Loads the signal/feature definitions from src/tremor_simulation.py WITHOUT running its experiment.

The research scripts execute their whole experiment at import time (train, evaluate, write JSON), so they cannot be
imported by a test. Rather than restructure published research code, this executes only the top-level statements that
come before the line `R={}` (imports, constants and function definitions) in an isolated namespace.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "tremor_simulation.py"


def _load_definitions(path):
    tree = ast.parse(path.read_text())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "R" for t in node.targets):
            break                                           # everything after this is the experiment itself
        keep.append(node)
    ns = {"__name__": "tremor_defs"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    return ns


@pytest.fixture(scope="session")
def sim():
    return _load_definitions(SRC)


@pytest.fixture(scope="session")
def results():
    import json
    return {p.stem: json.loads(p.read_text()) for p in (ROOT / "results").glob("*.json")}


@pytest.fixture(scope="session")
def readme():
    return (ROOT / "README.md").read_text()
