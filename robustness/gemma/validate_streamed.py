"""Check frozen seed selection, cooperative ownership, and CPU master restoration."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bundle import sha
from gemma import adapter, streamed_cohort
from gemma.streamed_mlp import StreamedMLP
from runtime import RUNS, configure


class SelectionReady(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure("gemma")
    import model_io
    import seed_evaluate
    from model_io import setup

    setup()
    checks = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source, out = RUNS / "gemma/divide", root / "gemma/divide"
        out.mkdir(parents=True)
        selected = json.loads((source / "selection.json").read_text())
        names = {
            "selection.json",
            "problems.json",
            "extra-problems.json",
            "component.pt",
            "component-active.pt",
            *selected["directories"],
        }
        names.update(
            spec["checkpoint"]
            for category in ["sparse", "linear"]
            for spec in selected[category].values()
        )
        for name in names:
            (out / name).symlink_to(source / name, target_is_directory=(source / name).is_dir())

        def stop_before_model():
            raise SelectionReady()

        def configure_selection(model):
            configure(model)
            model_io.load = stop_before_model

        with (
            patch.object(seed_evaluate, "RUNS", root),
            patch.object(seed_evaluate, "configure", configure_selection),
            patch.object(sys, "argv", ["seed_evaluate", "gemma", "divide"]),
        ):
            try:
                seed_evaluate.main()
            except SelectionReady:
                pass
            else:
                raise AssertionError("Frozen stage did not stop before model loading")
        expected = streamed_cohort.seed_selection(source)
        assert json.loads((out / "all-seeds-selection.json").read_text()) == expected
        selection_hash = sha(out / "all-seeds-selection.json")
        checks.append("selection equals the original frozen seed stage")

        (root / "queue-gemma.json").write_text(
            json.dumps({"status": "running", "child_pid": os.getpid()})
        )
        with patch.object(streamed_cohort, "RUNS", root):
            for command, ordinary_exists, owned in [
                (["seed_evaluate", "gemma", "divide"], False, False),
                (["seed_evaluate", "gemma", "divide"], True, True),
                (["geometry_evaluate", "gemma", "divide", "--validation-only"], True, False),
                (["geometry_evaluate", "gemma", "divide"], True, True),
            ]:
                marker = out / "ordinary-all-seeds.json"
                if ordinary_exists:
                    marker.write_text("{}")
                else:
                    marker.unlink(missing_ok=True)
                child = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(30)", *command]
                )
                try:
                    assert streamed_cohort.main_owns_cohort() == owned
                finally:
                    child.terminate()
                    child.wait()
        checks.append(
            "ownership distinguishes answer generation, interventions, and preliminary geometry"
        )

    class Broken(torch.nn.Linear):
        def forward(self, x):
            raise RuntimeError("intentional forward failure")

    mlp = StreamedMLP(Broken(2, 2)).eval()
    before = [master.detach().clone() for _, _, master in mlp.slots]
    with torch.inference_mode():
        try:
            mlp(torch.ones(1, 2, device="cuda"))
        except RuntimeError as error:
            assert str(error) == "intentional forward failure"
        else:
            raise AssertionError("Failed forward unexpectedly returned")
    torch.cuda.synchronize()
    for (module, name, master), original in zip(mlp.slots, before):
        assert module._parameters[name] is master and master.device.type == "cpu"
        assert master.is_pinned() and torch.equal(master, original)
    checks.append("failed CUDA forward restores unchanged pinned CPU parameters")
    proof = {
        "passed": True,
        "checks": checks,
        "frozen_seed_selection_sha256": selection_hash,
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": {
            name: sha(adapter.STUDY / name)
            for name in [
                "gemma/validate_streamed.py",
                "gemma/streamed_mlp.py",
                "gemma/streamed_cohort.py",
                "seed_evaluate.py",
            ]
        },
    }
    args.output.write_text(json.dumps(proof, indent=2) + "\n")
    print("STREAMED EXECUTION CONTROLS PASSED", len(checks), flush=True)


if __name__ == "__main__":
    main()
