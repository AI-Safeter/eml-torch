"""Smoke test for the shipped README figures.

These figures are referenced by the README at the top of the repo. The
test runs the figure-generation scripts end-to-end and asserts each
produced PNG is larger than a minimum-realistic size, which is the
sanity bound that catches silent-failure modes like the matplotlib
all-NaN ``imshow`` bug we hit in commit 9eb4523 (the heat-strip panel
rendered as an empty axes — file was ~30 kB instead of ~150 kB).

Skips if matplotlib is not installed in the test environment.

The scripts run on CPU only with reduced budgets where possible; they
should each complete in well under 60 seconds.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EX_DIR = REPO_ROOT / "examples" / "srbench_feynman"


def _matplotlib_available() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


pytestmark = pytest.mark.skipif(
    not _matplotlib_available(), reason="matplotlib not installed"
)


def _run_script(script_name: str) -> None:
    """Invoke a figure-generation script as a subprocess.

    Subprocess isolation matters here: the scripts call matplotlib.use("Agg")
    at import time, which is a process-global toggle — running them in-process
    after another test has set a different backend would either fail loudly
    or (worse) silently re-route the output to an interactive backend.
    """
    script = EX_DIR / script_name
    assert script.exists(), f"Missing example script: {script}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run(
        [sys.executable, "-u", str(script)],
        cwd=str(EX_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert res.returncode == 0, (
        f"{script_name} exited {res.returncode}.\n"
        f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )


# ---------------------------------------------------------------------------
# benchmark + pareto demo: regenerates two figures from the shipped JSON
# ---------------------------------------------------------------------------


def test_make_figures_produces_nonempty_pngs():
    """make_figures.py emits figure_benchmark_v2.png and figure_pareto_demo_v2.png.

    Min-size threshold = 50 kB. An empty/all-NaN heat-strip PNG is ~30 kB
    (this exact bug shipped briefly before the fix in 9eb4523).
    """
    bench = EX_DIR / "figure_benchmark_v2.png"
    pareto = EX_DIR / "figure_pareto_demo_v2.png"
    # Remove any stale outputs so we know the script wrote them this run.
    for p in (bench, pareto):
        if p.exists():
            p.unlink()

    _run_script("make_figures.py")

    for p in (bench, pareto):
        assert p.exists(), f"{p.name} was not produced"
        size = p.stat().st_size
        assert size > 50_000, (
            f"{p.name} is suspiciously small ({size} bytes) — possible "
            "silent-failure render (e.g. all-NaN imshow). Open the file to inspect."
        )


# ---------------------------------------------------------------------------
# exp(a·b) demo: runs an actual depth-1 EML evolution with use_mul=True
# ---------------------------------------------------------------------------


def test_make_eml_wins_figure_produces_nonempty_png():
    """make_eml_wins_figure.py emits figure_eml_wins_v2.png.

    Same min-size guard. Also serves as a smoke test that depth-1
    ``use_mul=True`` evolution still finds ``eml((x1*x2), 1)`` on
    ``exp(x1*x2)`` — if the structural recovery regresses, the figure
    would still be produced but downstream R² assertions could be
    tightened in a follow-up test.
    """
    out = EX_DIR / "figure_eml_wins_v2.png"
    if out.exists():
        out.unlink()

    _run_script("make_eml_wins_figure.py")

    assert out.exists(), "figure_eml_wins_v2.png was not produced"
    size = out.stat().st_size
    assert size > 50_000, (
        f"figure_eml_wins_v2.png is suspiciously small ({size} bytes) — "
        "possible silent-failure render."
    )
