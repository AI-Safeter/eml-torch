"""Training-only PLS feature refinement after poor PCA validation, before test."""

import shutil

import torch
from common import OUT, dump, load, setup


def main():
    setup()
    model, _ = load()
    state = torch.load(OUT / "component.pt", weights_only=True)
    data = torch.load(OUT / "component-data.pt", weights_only=True)
    h = data["train"]["input"].cuda()
    d = state["direction"].cuda()
    mean = state["input_mean"].cuda()
    layer = model.model.layers[state["layer"]].mlp
    with torch.inference_mode():
        hs = []
        ys = []
        for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
            mix = h[:, 1] * (1 - a) + h[:, 0] * a
            hs.append(mix - mean)
            ys.append(torch.cat([layer(x) @ d for x in mix.split(128)]))
        x = torch.cat(hs).double()
        y = torch.cat(ys).double()
        y -= y.mean()
        xr = x.clone()
        yr = y.clone()
        ws = []
        ps = []
        for _ in range(3):
            w = xr.T @ yr
            w /= w.norm()
            t = xr @ w
            p = (xr.T @ t) / (t @ t)
            q = (yr @ t) / (t @ t)
            xr -= t[:, None] * p
            yr -= t * q
            ws.append(w)
            ps.append(p)
        w = torch.stack(ws, 1)
        p = torch.stack(ps, 1)
        encoder = (w @ torch.linalg.inv(p.T @ w)).float()
    archive = OUT / "pca-attempt"
    archive.mkdir(exist_ok=True)
    for name in [
        "component.pt",
        "replacements.pt",
        "fits.json",
        "candidates.json",
        "fit-protocol.json",
        "fit-data.pt",
        "eml-observed.pt",
        "eml-intervention.pt",
        "fit.log",
    ]:
        if (OUT / name).exists():
            shutil.copy2(OUT / name, archive / name)
    state["encoder"] = encoder.cpu()
    state["feature_method"] = (
        "three-component supervised partial least squares, training mixtures only"
    )
    torch.save(state, OUT / "component.pt")
    dump(
        "feature-amendment.json",
        {
            "status": "before any test evaluation",
            "reason": "PCA-based validation normalized MSE 0.657 for EML and 0.432 for neural control suggests inadequate coordinates",
            "method": state["feature_method"],
            "target": "same selected MLP-output coefficient, no numerical answer or downstream logit input",
            "original_attempt": "pca-attempt/",
            "component_selection_unchanged": True,
            "rank": 3,
            "encoder_parameters": encoder.numel(),
            "train_only": True,
        },
    )
    print("PLS training features written", flush=True)


if __name__ == "__main__":
    main()
