"""TDD tests for the eml-vacuity-audit CLI (no model load: scores-JSON mode).

Run (CPU, from emltorch/):
    CUDA_VISIBLE_DEVICES="" pytest tests/test_vacuity_cli.py -q
"""

from __future__ import annotations

import json
import math

from emltorch.certify.vacuity_cli import main


def _write(tmp_path, blob):
    p = tmp_path / "head.json"
    p.write_text(json.dumps(blob))
    return str(p)


def _logprob_head(p, n=4):
    return [math.log(p)] + [math.log((1 - p) / (n - 1))] * (n - 1)


def test_cli_requires_an_input_source(capsys):
    rc = main(["--tau", "0.95"])
    assert rc == 2
    assert "supply" in capsys.readouterr().err


def test_cli_vacuous_v3_logprob_returns_nonzero(tmp_path, capsys):
    path = _write(tmp_path, {"scores": _logprob_head(0.10), "target_idx": 0})
    rc = main(["--scores-json", path, "--tau", "0.95", "--form", "v3"])
    out = capsys.readouterr().out
    assert "VACUOUS" in out
    assert rc != 0  # not SOUND


def test_cli_sound_softmax_interval_returns_zero(tmp_path, capsys):
    path = _write(tmp_path, {"scores": _logprob_head(0.99), "target_idx": 0})
    rc = main(
        [
            "--scores-json",
            path,
            "--tau",
            "0.95",
            "--form",
            "softmax_interval",
            "--precision-floor",
            "0.001",
        ]
    )
    out = capsys.readouterr().out
    assert "SOUND" in out
    assert rc == 0


def test_cli_json_mode_emits_parseable_report(tmp_path, capsys):
    path = _write(tmp_path, {"scores": _logprob_head(0.99), "target_idx": 0})
    rc = main(["--scores-json", path, "--tau", "0.95", "--json"])
    blob = json.loads(capsys.readouterr().out)
    assert isinstance(blob, list) and len(blob) == 1
    assert blob[0]["verdict"] == "SOUND"
    assert rc == 0


def test_cli_atlas_list_summary(tmp_path, capsys):
    rows = [
        {"scores": _logprob_head(0.99), "target_idx": 0, "layer": 5, "head": 5},
        {"scores": _logprob_head(0.10), "target_idx": 0, "layer": 0, "head": 0},
    ]
    path = _write(tmp_path, rows)
    rc = main(["--scores-json", path, "--tau", "0.95", "--form", "softmax_interval"])
    out = capsys.readouterr().out
    assert "summary:" in out
    # one of two heads is SOUND -> overall non-zero
    assert rc != 0


def test_cli_raw_weight_relative_only(tmp_path, capsys):
    # BOS (idx0) holds 91% of mass, target (idx2) 9%; excluding BOS+self makes
    # the cert discharge -> RELATIVE-ONLY.
    rows = {
        "scores": [50.0, 0.02, 5.0, 0.02, 0.01, 0.05],
        "target_idx": 2,
        "exclude_from_sum": [0, 5],
    }
    path = _write(tmp_path, rows)
    rc = main(["--scores-json", path, "--tau", "0.95", "--form", "raw_weight"])
    out = capsys.readouterr().out
    assert "RELATIVE-ONLY" in out
    assert rc != 0
