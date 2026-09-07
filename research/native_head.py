"""Pack frozen scalar heads for a small CUDA inference kernel."""

import ctypes
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


class NativeHead:
    def __init__(self, head, component, split_projection=False):
        source, library = ROOT / "head_kernel.cu", ROOT / "head_kernel.so"
        if not library.exists() or source.stat().st_mtime > library.stat().st_mtime:
            subprocess.run(
                [
                    "nvcc",
                    "-O3",
                    "--shared",
                    "-Xcompiler",
                    "-fPIC",
                    "-arch=sm_90",
                    str(source),
                    "-o",
                    str(library),
                ],
                check=True,
            )
        self.library = ctypes.CDLL(str(library))
        self.split_projection = split_projection
        self.launch = (
            self.library.launch_head_split if split_projection else self.library.launch_head
        )
        self.launch.argtypes = (
            [ctypes.c_void_p] * (13 if split_projection else 12)
            + [ctypes.c_int] * 5
            + [ctypes.c_float] * 2
            + [ctypes.c_void_p]
        )
        self.launch.restype = ctypes.c_int
        self.kind = {
            "eml_square": 0,
            "eml_log": 1,
            "eml_affine": 2,
            "silu": 3,
            "silu_two": 4,
        }[head.kind]
        self.rank, self.width = (
            head.rank,
            head.width * (2 if head.kind == "silu" else 1),
        )
        rank, width = self.rank, self.width
        if not 1 <= rank <= 32 or not 1 <= width <= 64:
            raise ValueError("Native head supports ranks 1..32 and effective widths 1..64")
        self.dimension = len(component["input_mean"])
        # Fold normalization in float64, then store float32 inference coefficients.
        encoder = component["encoder"][:, :rank].double().cuda()
        mean = component["input_mean"].double().cuda()
        std, zmean = (
            component["zstd"][:rank].double().cuda(),
            component["zmean"][:rank].double().cuda(),
        )
        projection = (encoder / std).T.contiguous().float()
        bias = (-(mean @ encoder + zmean) / std).float()
        state = head.state_dict()
        skip = torch.cat([state["skip.weight"].flatten(), state["skip.bias"]]).cuda()
        zeros = torch.zeros(width, rank, device="cuda")
        zero_bias = torch.zeros(width, device="cuda")
        left = state.get("left.weight", state.get("hidden.weight", zeros))
        left_bias = state.get("left.bias", state.get("hidden.bias", zero_bias))
        right = state.get("right.weight", zeros)
        right_bias = state.get("right.bias", zero_bias)
        second = state.get("hidden2.weight", torch.zeros(width, width, device="cuda"))
        second_bias = state.get("hidden2.bias", zero_bias)
        self.tensors = [
            v.cuda().float().contiguous()
            for v in [
                projection,
                bias,
                skip,
                left,
                left_bias,
                right,
                right_bias,
                second,
                second_bias,
                state["out.weight"].flatten(),
            ]
        ]
        self.mean, self.scale = float(component["ymean"]), float(component["ystd"])

    def __call__(self, x, out=None):
        if torch.is_grad_enabled() and x.requires_grad:
            raise ValueError("The native head supports inference only")
        if x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous():
            raise ValueError("Expected contiguous CUDA float32 inputs")
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError("Input dimensions do not match the exported head")
        if torch.cuda.current_device() != x.device.index:
            raise ValueError("Input must be on the current CUDA device")
        if x.device != self.tensors[0].device:
            raise ValueError("Input and exported coefficients must be on the same CUDA device")
        if out is None:
            out = torch.empty(len(x), device=x.device, dtype=torch.float32)
        if (
            out.shape != (len(x),)
            or out.dtype != torch.float32
            or out.device != x.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "Output must be contiguous CUDA float32 with one element per input row"
            )
        if len(x) == 0:
            return out
        if out.requires_grad:
            raise ValueError("The native output buffer must not require gradients")
        if any(
            out.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
            for v in [x, *self.tensors]
        ):
            raise ValueError("Output must not alias inputs or exported coefficients")
        projected = (
            torch.empty(len(x), self.rank, device=x.device, dtype=torch.float32)
            if self.split_projection
            else None
        )
        status = self.launch(
            x.data_ptr(),
            *[v.data_ptr() for v in self.tensors],
            out.data_ptr(),
            *([projected.data_ptr()] if projected is not None else []),
            len(x),
            self.dimension,
            self.rank,
            self.width,
            self.kind,
            self.mean,
            self.scale,
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(f"CUDA head launch failed: {status}")
        return out
