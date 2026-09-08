"""Verify that raw traces contain the frozen problems, edits, and controls exactly."""

import math
from collections import Counter
from itertools import product


def identity(row):
    return tuple(
        row.get(key)
        for key in [
            "a",
            "b",
            "c",
            "corrupt_b",
            "answer",
            "corrupt_answer",
            "op",
            "style",
            "alpha",
            "feature",
            "geometry",
        ]
    )


def check(data, rows, kinds, *, alphas=None, features=None, geometry=None):
    assert set(data) == set(kinds), (set(data) - set(kinds), set(kinds) - set(data))
    expected = Counter(
        identity({**row, "alpha": alpha, "feature": feature, "geometry": geometry})
        for row, alpha, feature in product(rows, alphas or [None], features or [None])
    )
    assert all(count == 1 for count in expected.values()), "Duplicate frozen condition"
    baseline = {identity(row): row for row in data["original"]}
    for kind, records in data.items():
        assert Counter(map(identity, records)) == expected, f"Missing/duplicate conditions: {kind}"
        for row in records:
            assert all(
                math.isfinite(value) for value in row.values() if isinstance(value, float)
            ), f"Nonfinite trace value: {kind}"
            original = baseline[identity(row)]
            assert row["true_coefficient"] == original["true_coefficient"], kind
            if alphas is None:
                assert row["base_text"] == original["text"]
                assert row["base_correct"] == original["correct"]
                assert row["exact_text_agreement"] == (row["text"] == original["text"])
                if kind == "restore":
                    assert row["text"] == original["text"] and row["kl"] == 0
            elif kind == "restore":
                assert row["margin"] == original["margin"]
                assert row["kl_vs_same_intervention_original"] == 0
    return {"methods": len(data), "unique_conditions_per_method": len(expected)}
