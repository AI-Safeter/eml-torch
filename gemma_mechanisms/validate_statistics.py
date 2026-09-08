"""CUDA checks of grouping and conservative retention bounds using labeled fixtures."""

import json

from .quality_summary import accuracy, bootstrap
from .runtime import HERE, digest, save, setup


def main():
    setup(73)
    alpha = 0.05 / 15
    teacher = [
        {"id": str(i), "style": style, "correct": True} for i in range(1024) for style in ["a", "b"]
    ]
    same = accuracy(teacher, teacher, alpha, True)
    assert same["groups"] == 1024 and same["regressing_groups"] == 0
    assert abs(same["loss_upper_bound"] - (1 - alpha ** (1 / 1024))) < 1e-12
    student = [dict(r) for r in teacher]
    student[0]["correct"] = False
    changed = accuracy(teacher, student, alpha, True)
    assert changed["regressing_groups"] == 1
    assert changed["net_loss"] == 1 / 2048
    assert changed["loss_upper_bound"] > changed["net_loss"]
    try:
        accuracy(teacher + teacher[:1], teacher + teacher[:1], alpha)
    except AssertionError:
        pass
    else:
        raise AssertionError("Duplicated identities accepted")
    zero = bootstrap([0.0] * 512, alpha)
    assert zero["mean"] == zero["upper"] == 0
    constant = bootstrap([0.02] * 512, alpha)
    assert abs(constant["upper"] - 0.02) < 1e-12
    save(
        HERE / "statistics-validation.json",
        {
            "fixtures_not_scientific_results": True,
            "checks": [
                "Formats grouped by operand identity",
                "Exact zero-regression binomial bound",
                "Regression upper bound distinguished from net loss",
                "Duplicate identities rejected",
                "CUDA bootstrap preserves constant contrasts",
            ],
            "source_sha256": digest(HERE / "quality_summary.py"),
            "alpha": alpha,
        },
    )
    print(json.dumps({"checks": "passed", "zero_regression_upper": same["loss_upper_bound"]}))


if __name__ == "__main__":
    main()
