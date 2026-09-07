"""Fold preprocessing and pack affine arguments for ordinary Torch compilation."""

import torch
from torch import nn

from emltorch.operator import safe_eml


class PackedHead(nn.Module):
    def __init__(self, head, component):
        super().__init__()
        self.kind, self.width = head.kind, head.width
        rank = head.rank
        encoder = component["encoder"][:, :rank].double().cuda()
        std = component["zstd"][:rank].double().cuda()
        bias = (
            -(
                component["input_mean"].double().cuda() @ encoder
                + component["zmean"][:rank].double().cuda()
            )
            / std
        )
        self.register_buffer("projection", (encoder / std).T.contiguous().float())
        self.register_buffer("projection_bias", bias.float())
        if self.kind in ["eml_square", "eml_affine"]:
            linears = [head.left, head.right, head.skip]
        elif self.kind == "eml_log":
            linears = [head.right, head.skip]
        else:
            linears = [head.hidden, head.skip]
        self.register_buffer("weight", torch.cat([layer.weight.detach() for layer in linears]))
        self.register_buffer("bias", torch.cat([layer.bias.detach() for layer in linears]))
        self.register_buffer("readout", head.out.weight.detach().flatten())
        if self.kind == "silu_two":
            self.register_buffer("second_weight", head.hidden2.weight.detach())
            self.register_buffer("second_bias", head.hidden2.bias.detach())
        self.register_buffer("output_mean", component["ymean"].cuda())
        self.register_buffer("output_scale", component["ystd"].cuda())

    def forward(self, x):
        z = nn.functional.linear(x, self.projection, self.projection_bias)
        values = nn.functional.linear(z, self.weight, self.bias)
        skip = values[:, -1]
        if self.kind in ["eml_square", "eml_affine"]:
            left, right = (
                values[:, : self.width],
                values[:, self.width : 2 * self.width],
            )
            hidden = safe_eml(left, 1 + right.square() if self.kind == "eml_square" else right)
        elif self.kind == "eml_log":
            right = values[:, : self.width]
            hidden = safe_eml(torch.zeros_like(right), 1 + right.square())
        else:
            hidden = nn.functional.silu(values[:, :-1])
            if self.kind == "silu_two":
                hidden = nn.functional.silu(
                    nn.functional.linear(hidden, self.second_weight, self.second_bias)
                )
        return (skip + (hidden * self.readout).sum(1)) * self.output_scale + self.output_mean
