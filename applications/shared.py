"""GPU fitting, held-out metrics, and dependency-free equation deployment."""

import copy
import importlib.util
import itertools
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

import emltorch
from emltorch._ast import _EML, _Add, _Const, _Div, _Exp, _Mul, _parse_inner, _Sub, _Var
from emltorch.operator import safe_eml

ROOT = Path(__file__).resolve().parent


def setup(seed=20260907):
    if not torch.cuda.is_available():
        raise RuntimeError("These validation runs require a CUDA GPU")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "seed": seed,
        "tf32": False,
    }


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def metrics(pred, target):
    pred, target = pred.double(), target.double()
    if not torch.isfinite(pred).all():
        return {
            "rmse": None,
            "mae": None,
            "r2": None,
            "max_error": None,
            "finite_fraction": float(torch.isfinite(pred).double().mean()),
            "status": "failed: nonfinite predictions",
        }
    mse = (pred - target).square().mean()
    variance = (target - target.mean()).square().mean().clamp_min(1e-20)
    return {
        "rmse": float(mse.sqrt()),
        "mae": float((pred - target).abs().mean()),
        "r2": float(1 - mse / variance),
        "max_error": float((pred - target).abs().max()),
    }


def expression_source(node, backend="torch"):
    if isinstance(node, _Const):
        return repr(node.value)
    if isinstance(node, _Var):
        if not node.name.startswith("x") or not node.name[1:].isdigit():
            raise ValueError("Unexpected variable")
        return f"x[..., {int(node.name[1:])}]" if backend == "torch" else f"x[{int(node.name[1:])}]"
    if isinstance(node, _EML):
        return (
            f"E({expression_source(node.left, backend)}, {expression_source(node.right, backend)})"
        )
    if isinstance(node, _Exp):
        return f"EXP({expression_source(node.arg, backend)})"
    op = {_Add: "+", _Sub: "-", _Mul: "*", _Div: "/"}[type(node)]
    return (
        f"({expression_source(node.left, backend)} {op} {expression_source(node.right, backend)})"
    )


def formula_function(expression):
    source = expression_source(_parse_inner(expression))

    def predict(x):
        def E(a, b):
            a = torch.as_tensor(a, dtype=x.dtype, device=x.device)
            b = torch.as_tensor(b, dtype=x.dtype, device=x.device)
            return safe_eml(a, b)

        return eval(code, {"__builtins__": {}, "E": E, "EXP": torch.exp}, {"x": x})

    code = compile(source, "<validated-expression>", "eval")
    return predict


def fit_equation(x, y, xv, yv, out, rounds=4):
    """Fit residual stages on training only; select stages and candidates by validation."""
    start = time.perf_counter()
    bias = y.mean()
    train_pred, val_pred = torch.full_like(y, bias), torch.full_like(yv, bias)
    accepted, candidates, stages = [], [], []
    best_error = float((val_pred - yv).square().mean())
    for stage in range(rounds):
        choices = []
        for depth, seed in itertools.product([2, 3, 4], [17, 29]):
            torch.manual_seed(seed + stage * 100)
            fit = emltorch.fit(
                x,
                y - train_pred,
                depth=depth,
                strategy="evolution",
                population=1024,
                generations=80,
                device="cuda",
                use_mul=True,
                polish=True,
                polish_iters=200,
                polish_optimizer="lbfgs",
                r2_target=0.9999,
                var_names=[f"x{i}" for i in range(x.shape[1])],
            )
            fn = formula_function(fit.expression)
            pt, pv = fn(x), fn(xv)
            reference = fit.predict(xv).cuda()
            assert torch.allclose(pv, reference, atol=2e-4, rtol=2e-4), "Export differs from fit"
            error = float((val_pred + pv - yv).square().mean())
            row = {
                "stage": stage,
                "depth": depth,
                "seed": seed,
                "validation_mse": error,
                "expression": fit.expression,
                "eml_nodes": fit.expression.count("eml("),
                "fit_seconds": fit.time_s,
            }
            candidates.append(row)
            choices.append((error, row, pt, pv))
        error, row, pt, pv = min(choices, key=lambda a: a[0])
        if error >= best_error:
            break
        best_error = error
        accepted.append(row["expression"])
        train_pred, val_pred = train_pred + pt, val_pred + pv
        stages.append({"stage": stage, "validation_mse": error, "eml_nodes": row["eml_nodes"]})
        print("EML stage", stage, "validation MSE", error, flush=True)
    model = {
        "bias": float(bias),
        "expressions": accepted,
        "stages": stages,
        "search_seconds": time.perf_counter() - start,
    }
    save(out / "search.json", candidates)
    return model


def predict_equation(model, x):
    result = x.new_full((len(x),), model["bias"])
    for expression in model["expressions"]:
        result = result + formula_function(expression)(x)
    return result


def train_network(x, y, xv, yv):
    selected = None
    records = []
    for seed in [17, 29, 43]:
        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 1),
        ).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
        best, checkpoint, stale = math.inf, None, 0
        for step in range(2500):
            opt.zero_grad()
            loss = (model(x).squeeze(1) - y).square().mean()
            loss.backward()
            opt.step()
            if step % 25 == 0:
                with torch.no_grad():
                    value = float((model(xv).squeeze(1) - yv).square().mean())
                if value < best:
                    best, checkpoint, stale = value, copy.deepcopy(model.state_dict()), 0
                else:
                    stale += 1
                if stale >= 16:
                    break
        model.load_state_dict(checkpoint)
        model.eval().requires_grad_(False)
        records.append({"seed": seed, "validation_mse": best, "steps": step + 1})
        if selected is None or best < selected[0]:
            selected = best, model
    return selected[1], records


def design(x, kind):
    columns = [torch.ones(len(x), device=x.device, dtype=x.dtype)]
    degree = int(kind[-1]) if kind.startswith("polynomial") else 3
    if kind == "spline":
        for i in range(x.shape[1]):
            v = x[:, i]
            columns.extend([v, v**2, v**3])
            columns.extend(torch.relu(v - k) ** 3 for k in [-1.5, -0.75, 0.0, 0.75, 1.5])
    else:
        for d in range(1, degree + 1):
            columns.extend(
                x[:, list(idx)].prod(1)
                for idx in itertools.combinations_with_replacement(range(x.shape[1]), d)
            )
    return torch.stack(columns, 1)


def fit_baselines(x, y, xv, yv):
    models = {}
    for kind in ["polynomial1", "polynomial2", "polynomial3", "spline"]:
        a, b = design(x.double(), kind), design(xv.double(), kind)
        best = None
        for penalty in [1e-6, 1e-4, 0.01, 1.0]:
            regularizer = torch.eye(a.shape[1], device="cuda", dtype=torch.float64) * penalty
            regularizer[0, 0] = 0
            w = torch.linalg.solve(a.T @ a + regularizer, a.T @ y.double())
            error = float((b @ w - yv).square().mean())
            if best is None or error < best[0]:
                best = error, w, penalty
        models[kind] = {"weights": best[1].float(), "penalty": best[2], "validation_mse": best[0]}
    return models


def timed(fn, repeats=100):
    for _ in range(5):
        fn()
    values = []
    for _ in range(repeats):
        t = time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns() - t) / 1000)
    return {
        "median_us": float(np.median(values)),
        "p90_us": float(np.quantile(values, 0.9)),
        "repeats": repeats,
    }


def deploy(model, mean, std, ymean, ystd, inputs, path, bounds, energy_prior=False):
    """Export validated expressions to a Python module using only the standard library."""
    expressions = [expression_source(_parse_inner(e), "scalar") for e in model["expressions"]]
    source = '''"""Generated EML surrogate. Inputs outside the documented range are unvalidated."""
import math

def E(a, b):
    return math.exp(max(-80., min(80., a))) - math.log(max(1e-6, min(1e30, b)))

EXP = math.exp
'''
    source += f"\nMEAN = {mean.tolist()!r}\nSTD = {std.tolist()!r}\n"
    source += f"LOW = {bounds[0].tolist()!r}\nHIGH = {bounds[1].tolist()!r}\n"
    source += f"\ndef predict_one(features):\n    if len(features) != {len(mean)}:\n        raise ValueError('Wrong feature count')\n"
    source += "    if not all(math.isfinite(v) for v in features):\n        raise ValueError('Inputs must be finite')\n"
    source += "    if any(v < lo or v > hi for v,lo,hi in zip(features, LOW, HIGH)):\n        raise ValueError('Outside the evaluated input range')\n"
    source += "    x = [(v-m)/s for v,m,s in zip(features, MEAN, STD)]\n"
    source += f"    value = {model['bias']!r}\n"
    source += "".join(f"    value += {e}\n" for e in expressions)
    source += f"    value = value * {float(ystd)!r} + {float(ymean)!r}\n"
    source += (
        "    return (1-math.cos(features[0])) * math.exp(value)\n"
        if energy_prior
        else "    return value\n"
    )
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("deployed_equation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # GPU comparison uses the production float32 equations against scalar float64 export.
    within = ((inputs >= bounds[0]) & (inputs <= bounds[1])).all(1)
    accepted_inputs = inputs[within]
    ref = predict_equation(model, (accepted_inputs - mean) / std) * ystd + ymean
    if energy_prior:
        ref = (1 - torch.cos(accepted_inputs[:, 0])) * torch.exp(ref)
    actual = torch.tensor(
        [module.predict_one(r) for r in accepted_inputs.cpu().tolist()], device="cuda"
    )
    assert torch.isfinite(actual).all()
    max_error = float((actual - ref).abs().max())
    assert torch.allclose(actual, ref, atol=max(0.002 * float(ystd), 0.002), rtol=2e-4), max_error
    audit = {
        "max_export_error": max_error,
        "bytes": path.stat().st_size,
        "range_accepted": int(within.sum()),
        "range_rejected": int((~within).sum()),
        "range_warning": "A bounding box does not guarantee accuracy or detect every distribution shift.",
    }
    if (~within).any():
        try:
            module.predict_one(inputs[~within][0].cpu().tolist())
        except ValueError:
            audit["range_guard_verified"] = True
        else:
            raise AssertionError("Range guard did not reject an out-of-range input")
    row = accepted_inputs[0].cpu().tolist()
    audit["cpu_single"] = timed(lambda: module.predict_one(row))
    return audit
