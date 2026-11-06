"""Load the eval scripts, which live outside the package, so tests can import their functions."""

import importlib.util
from pathlib import Path

EVALS = Path(__file__).resolve().parents[1] / "evals"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EVALS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score_ranking = _load("retrieval_eval").score_ranking
answer_eval = _load("answer_eval")
