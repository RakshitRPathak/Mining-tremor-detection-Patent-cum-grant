"""Stage 1 claims 'fixed random seeds make every number reproducible'. This re-runs it and compares to the committed JSON."""
import json
import shutil
import subprocess
import sys

import pytest

from conftest import ROOT


def _differences(a, b, path=""):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), f"keys differ at {path or '/'}"
        for k in a:
            yield from _differences(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"length differs at {path}"
        for i, (x, y) in enumerate(zip(a, b)):
            yield from _differences(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        if abs(a - b) > 1e-9:
            yield path, a, b
    elif a != b:
        yield path, a, b


@pytest.mark.slow
def test_stage1_reproduces_the_committed_results_exactly(tmp_path):
    work = tmp_path / "repo"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    subprocess.run([sys.executable, "tremor_simulation.py"], cwd=work / "src", check=True, stdout=subprocess.DEVNULL, timeout=900)
    name = "tremor_simulation_results.json"
    old, new = json.loads((ROOT / "results" / name).read_text()), json.loads((work / "results" / name).read_text())
    for d in (old, new):
        d.pop("py_infer_ms", None)                                  # wall-clock timing differs on every machine
    assert list(_differences(old, new)) == []
