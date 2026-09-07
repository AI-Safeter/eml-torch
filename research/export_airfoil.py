"""Export the frozen airfoil head to Python/C and validate against CUDA predictions."""

import ctypes
import importlib.util
import json
import pprint
import re
import statistics
import subprocess
import time

import numpy as np
import torch

from data import ROOT, save
from model_io import setup

OUT = ROOT / "airfoil-replay"
SOURCE = ROOT / "airfoil-replay/source"


def c_array(name, tensor):
    def literal(value):
        value = format(float(value), ".9g")
        return value + (".0" if "." not in value and "e" not in value else "") + "f"

    values = [literal(value) for value in tensor.flatten().tolist()]
    return "static const float " + name + "[] = {" + ",".join(values) + "};\n"


def main():
    setup()
    selection = json.loads((OUT / "selection.json").read_text())["eml"]
    assert selection["kind"] == "eml_square"
    state = torch.load(OUT / (selection["name"] + ".pt"), weights_only=True, map_location="cpu")
    saved = torch.load(OUT / "predictions.pt", weights_only=True, map_location="cpu")
    teacher = saved["teacher_state"]
    array = np.loadtxt(SOURCE / "airfoil_self_noise.dat")
    inputs = np.ascontiguousarray(array[:, :5], dtype=np.float32)
    training = array[saved["splits"]["train"], :5]
    parameters = {
        "state": {k: v.tolist() for k, v in state.items()},
        "xmean": teacher["xmean"].tolist(),
        "xstd": teacher["xstd"].tolist(),
        "ymean": float(teacher["ymean"]),
        "ystd": float(teacher["ystd"]),
        "bounds": [training.min(0).tolist(), training.max(0).tolist()],
    }
    save(OUT / "portable-parameters.json", parameters)
    python_source = (
        '''"""Frozen airfoil surrogate. Inputs: frequency, angle, chord, velocity, thickness.

Training-range checks reject extrapolation; they do not certify prediction accuracy.
Dataset: UCI Airfoil Self-Noise, Brooks, Pope and Marcolini (1989), CC BY 4.0.
"""
import math

PARAMETERS = '''
        + pprint.pformat(parameters, width=100, sort_dicts=True)
        + """

def predict(frequency_hz, attack_angle_deg, chord_m, velocity_m_per_s, thickness_m):
    x = [float(v) for v in [frequency_hz, attack_angle_deg, chord_m, velocity_m_per_s, thickness_m]]
    p, state = PARAMETERS, PARAMETERS["state"]
    if not all(math.isfinite(v) for v in x):
        raise ValueError("Inputs must be finite")
    if any(v < lo - 5e-7*max(abs(lo),abs(hi),1e-12) or v > hi + 5e-7*max(abs(lo),abs(hi),1e-12)
            for v, lo, hi in zip(x, *p["bounds"])):
        raise ValueError("Outside the training range")
    z = [(v - m) / s for v, m, s in zip(x, p["xmean"], p["xstd"])]
    def affine(weights, bias):
        return sum(a * b for a, b in zip(weights, z)) + bias
    value = affine(state["skip.weight"][0], state["skip.bias"][0])
    for left, lb, right, rb, weight in zip(state["left.weight"], state["left.bias"],
            state["right.weight"], state["right.bias"], state["out.weight"][0]):
        a, b = affine(left, lb), affine(right, rb)
        value += weight * (math.exp(max(-80.0, min(80.0, a))) - math.log(min(1e30, 1 + b*b)))
    return p["ymean"] + p["ystd"] * value
"""
    )
    (OUT / "airfoil_equation.py").write_text(python_source)
    code = "#include <math.h>\n"
    for name, value in [
        ("xm", teacher["xmean"]),
        ("xs", teacher["xstd"]),
        ("ym", teacher["ymean"].reshape(1)),
        ("ys", teacher["ystd"].reshape(1)),
        ("ewl", state["left.weight"]),
        ("ebl", state["left.bias"]),
        ("ewr", state["right.weight"]),
        ("ebr", state["right.bias"]),
        ("ewo", state["out.weight"]),
        ("ews", state["skip.weight"]),
        ("ebs", state["skip.bias"]),
    ]:
        code += c_array(name, value)
    for layer in [0, 2, 4]:
        code += c_array(f"nw{layer}", teacher["state"][f"{layer}.weight"])
        code += c_array(f"nb{layer}", teacher["state"][f"{layer}.bias"])
    width = selection["width"]
    code += """
static float dot(const float *w, const float *x, int n, float b) {
    for (int i=0;i<n;i++) b=fmaf(w[i],x[i],b);
    return b;
}
static float silu(float x) {return x/(1.0f+expf(-x));}
float airfoil_eml(const float *input) {
    float x[5];
    for(int j=0;j<5;j++) x[j]=(input[j]-xm[j])/xs[j];
    float value=dot(ews,x,5,ebs[0]);
    for(int i=0;i<WIDTH;i++) {
        float a=dot(ewl+i*5,x,5,ebl[i]),b=dot(ewr+i*5,x,5,ebr[i]);
        float unit=expf(fminf(80.0f,fmaxf(-80.0f,a)))-logf(fminf(1e30f,fmaf(b,b,1.0f)));
        value=fmaf(ewo[i],unit,value);
    }
    return fmaf(value,ys[0],ym[0]);
}
float airfoil_teacher(const float *input) {
    float x[5],h1[32],h2[32];
    for(int j=0;j<5;j++) x[j]=(input[j]-xm[j])/xs[j];
    for(int i=0;i<32;i++) h1[i]=silu(dot(nw0+i*5,x,5,nb0[i]));
    for(int i=0;i<32;i++) h2[i]=silu(dot(nw2+i*32,h1,32,nb2[i]));
    return fmaf(dot(nw4,h2,32,nb4[0]),ys[0],ym[0]);
}
void eml_batch(const float *input, float *output, int n) {
    for(int i=0;i<n;i++) output[i]=airfoil_eml(input+i*5);
}
void teacher_batch(const float *input, float *output, int n) {
    for(int i=0;i<n;i++) output[i]=airfoil_teacher(input+i*5);
}
""".replace("WIDTH", str(width))
    (OUT / "airfoil_models.c").write_text(code)
    eml_only = "\n".join(
        line
        for line in code.splitlines()
        if not line.startswith(("static const float nw", "static const float nb"))
    )
    eml_only = re.sub(r"float airfoil_teacher.*?(?=void eml_batch)", "", eml_only, flags=re.DOTALL)
    eml_only = re.sub(r"void teacher_batch.*", "", eml_only, flags=re.DOTALL)
    (OUT / "airfoil_equation.c").write_text(
        "/* Scalar EML predictor. Input bounds and units are in portable-parameters.json. */\n"
        + eml_only
    )
    subprocess.run(
        [
            "gcc",
            "-O3",
            "-march=native",
            "-shared",
            "-fPIC",
            str(OUT / "airfoil_equation.c"),
            "-lm",
            "-o",
            str(OUT / "airfoil_equation.so"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "gcc",
            "-O3",
            "-march=native",
            "-shared",
            "-fPIC",
            str(OUT / "airfoil_models.c"),
            "-lm",
            "-o",
            str(OUT / "airfoil_models.so"),
        ],
        check=True,
    )
    library = ctypes.CDLL(str(OUT / "airfoil_models.so"))
    pointer = ctypes.POINTER(ctypes.c_float)
    outputs = {}
    for label in ["eml", "teacher"]:
        function = getattr(library, label + "_batch")
        function.argtypes = [pointer, pointer, ctypes.c_int]
        function.restype = None
        output = np.empty(len(inputs), dtype=np.float32)
        function(inputs.ctypes.data_as(pointer), output.ctypes.data_as(pointer), len(inputs))
        outputs[label] = output
        expected = saved["predictions"]["eml"].cuda() if label == "eml" else saved["teacher"].cuda()
        actual = torch.tensor(output, device="cuda")
        torch.testing.assert_close(actual, expected, atol=0.001, rtol=1e-5)
    standalone = ctypes.CDLL(str(OUT / "airfoil_equation.so")).eml_batch
    standalone.argtypes, standalone.restype = [pointer, pointer, ctypes.c_int], None
    standalone_output = np.empty(len(inputs), dtype=np.float32)
    standalone(
        inputs.ctypes.data_as(pointer),
        standalone_output.ctypes.data_as(pointer),
        len(inputs),
    )
    torch.testing.assert_close(
        torch.tensor(standalone_output, device="cuda"),
        saved["predictions"]["eml"].cuda(),
        atol=0.001,
        rtol=1e-5,
    )
    spec = importlib.util.spec_from_file_location("portable_airfoil", OUT / "airfoil_equation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lo, hi = np.array(parameters["bounds"])
    tolerance = 5e-7 * np.maximum(np.maximum(np.abs(lo), np.abs(hi)), 1e-12)
    ids = np.flatnonzero(
        ((array[:, :5] >= lo - tolerance) & (array[:, :5] <= hi + tolerance)).all(1)
    ).tolist()
    assert set(saved["splits"]["train"]) <= set(ids)
    python_values = torch.tensor([module.predict(*array[i, :5]) for i in ids], device="cuda")
    expected = saved["predictions"]["eml"][ids].cuda()
    torch.testing.assert_close(python_values, expected, atol=0.001, rtol=1e-5)
    for invalid in [
        [float("nan"), 0, 0.1, 30, 0.01],
        array[saved["splits"]["shift"][0], :5].tolist(),
    ]:
        try:
            module.predict(*invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid or out-of-training-range input accepted")
    timings = {}
    many = np.tile(inputs, (8, 1))
    result = np.empty(len(many), dtype=np.float32)
    ptr = many.ctypes.data_as(pointer)
    for label in ["eml", "teacher"]:
        function = getattr(library, label + "_batch")
        scalar = getattr(library, "airfoil_" + label)
        scalar.argtypes, scalar.restype = [pointer], ctypes.c_float
        for _ in range(10):
            function(ptr, result.ctypes.data_as(pointer), len(many))
        amortized, call = [], []
        for _ in range(7):
            started = time.perf_counter()
            function(ptr, result.ctypes.data_as(pointer), len(many))
            amortized.append((time.perf_counter() - started) * 1e6 / len(many))
            started = time.perf_counter()
            for _ in range(5000):
                scalar(ptr)
            call.append((time.perf_counter() - started) * 1e6 / 5000)
        timings[label] = {
            "scalar_c_median_us": statistics.median(amortized),
            "python_ctypes_call_median_us": statistics.median(call),
            "scalar_c_repeats_us": amortized,
            "ctypes_repeats_us": call,
        }
    save(
        OUT / "portable-validation.json",
        {
            "completed_epoch": time.time(),
            "gpu_reference": torch.cuda.get_device_name(),
            "c_all_dataset_rows": len(inputs),
            "python_in_range_rows": len(ids),
            "c_max_absolute_error_vs_cuda": {
                label: float(
                    (
                        torch.tensor(values, device="cuda")
                        - (
                            saved["predictions"]["eml"].cuda()
                            if label == "eml"
                            else saved["teacher"].cuda()
                        )
                    )
                    .abs()
                    .max()
                )
                for label, values in outputs.items()
            },
            "python_max_absolute_error_vs_cuda": float((python_values - expected).abs().max()),
            "parameters": {"eml": selection["parameters"], "teacher": 1281},
            "timings": timings,
            "timing_scope": "Scalar CPU inference, both EML and the original teacher compiled to C with identical compiler flags. Amortized C loop and Python ctypes call overhead reported separately. Shared host.",
            "standalone_python_bytes": (OUT / "airfoil_equation.py").stat().st_size,
            "combined_c_library_bytes": (OUT / "airfoil_models.so").stat().st_size,
            "standalone_c_library_bytes": (OUT / "airfoil_equation.so").stat().st_size,
            "standalone_c_validated_rows": len(inputs),
            "bounds": "Python API rejects nonfinite inputs and values outside training ranges; C functions are unchecked benchmark primitives.",
            "source_attribution": "UCI Airfoil Self-Noise, Brooks, Pope and Marcolini (1989), CC BY 4.0; https://doi.org/10.24432/C5VW2C",
        },
    )
    print("PORTABLE AIRFOIL VALIDATED", timings, flush=True)


if __name__ == "__main__":
    main()
