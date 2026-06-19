"""CLI acceptance: eml-certify single-head and sweep modes run end-to-end on
GPT-2 (CPU) without error, and sweep mode writes valid JSON to --out.

Run from emltorch/:
    HF_HOME=/home/ubuntu/samuel/.cache/huggingface CUDA_VISIBLE_DEVICES="" \
        pytest tests/test_certify_cli.py -q
"""

from __future__ import annotations

import json

import pytest

from emltorch.certify.cli import main

# A short repeated prompt keeps the sweep fast on CPU.
PROMPT = "vase comet lunar drift vase comet lunar drift vase comet lunar"


@pytest.fixture(scope="module")
def _models_available():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return True


def test_single_head_mode_runs(_models_available, capsys):
    rc = main(
        [
            "--model",
            "gpt2",
            "--prompt",
            PROMPT,
            "--layer",
            "5",
            "--head",
            "5",
            "--tau",
            "0.5",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "L5.H5" in out or "layer" in out.lower()


def test_sweep_mode_writes_json(_models_available, tmp_path):
    out_path = tmp_path / "atlas.json"
    rc = main(
        [
            "--model",
            "gpt2",
            "--prompt",
            PROMPT,
            "--tau",
            "0.5",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    blob = json.loads(out_path.read_text())
    assert blob["model_name"] == "gpt2"
    assert blob["tau"] == 0.5
    assert len(blob["records"]) == 144
