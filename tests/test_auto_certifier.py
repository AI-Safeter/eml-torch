"""Tests for the AutoCertifier mechanistic interpretability pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from emltorch.interp import AutoCertifier


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 8)
        self.output_head = nn.Linear(8, 2)

    def forward(self, x=None, input_ids=None, **kwargs):
        if x is None:
            x = input_ids
        h = self.layer(x)

        class Outputs:
            def __init__(self, logits):
                self.logits = logits

        # Shape: (B, 1, 2) simulating decoder logits at sequence position
        logits = self.output_head(h).unsqueeze(1)
        return Outputs(logits)


def test_auto_certifier_pipeline():
    model = DummyModel()
    inputs = torch.randn(10, 10)
    targets = np.random.randn(10)

    # Initialize
    certifier = AutoCertifier(
        model=model,
        layer="layer",
        feature_reducer="pca",
        n_features=2,
        device="cpu",
    )

    # Fit
    model_eml = certifier.fit(
        inputs=inputs,
        targets=targets,
        depth=2,
        normalize_inputs=True,
        population=32,
        generations=3,
        r2_target=0.99,
    )

    assert certifier.model_result_ is not None
    assert certifier.X_reduced_ is not None
    assert certifier.X_reduced_.shape == (10, 2)

    # Generate SMT verification atlas
    with tempfile.TemporaryDirectory() as tmpdir:
        atlas = certifier.generate_verification_atlas(
            safety_threshold=10.0,
            properties=["bounds", "monotonicity", "lipschitz"],
            output_dir=tmpdir,
        )

        assert "bounds" in atlas
        assert "monotonicity" in atlas
        assert "lipschitz" in atlas

        assert Path(atlas["bounds"]).exists()
        assert Path(atlas["monotonicity"]).exists()
        assert Path(atlas["lipschitz"]).exists()

        # Simple verification that z3 can load them
        try:
            import z3
            for name, path in atlas.items():
                solver = z3.Solver()
                solver.from_file(str(path))
        except ImportError:
            pass
