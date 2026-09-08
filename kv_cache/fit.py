"""Fit GPU decoders without reading held-out documents or question scores."""

import copy

import torch
from common import RUN, SPEC, save_json, setup, sha
from v_codecs import Codec, parameter_elements, rank_for_budget


def main():
    setup()
    assert not (RUN / "codecs.pt").exists()
    values = torch.load(RUN / "values.pt", weights_only=True)
    saved = {m: [] for m in ["lowrank", "silu", "eml"]}
    rows = []
    for layer, (train, validation) in enumerate(zip(values["train"], values["validation"])):
        train, validation = train.cuda(), validation.cuda()
        mean = train.float().mean(0)
        centered = train.float() - mean
        _, vectors = torch.linalg.eigh(centered.T @ centered)
        d = train.shape[-1]
        tokens = SPEC["prefix_tokens"] if layer % 5 == 4 else 511
        for method in saved:
            torch.manual_seed(SPEC["seed"] + layer)
            r = rank_for_budget(method, d, tokens)
            basis = vectors[:, -r:].flip(-1).contiguous().bfloat16()
            codec = Codec(method, mean.bfloat16(), basis)
            ztrain, zval = codec.encode(train).detach(), codec.encode(validation).detach()
            best, best_step, best_state = float("inf"), 0, None
            optimizer = (
                None
                if method == "lowrank"
                else torch.optim.AdamW(
                    codec.head.parameters(), lr=SPEC["learning_rate"], weight_decay=1e-4
                )
            )
            for step in range(1 if optimizer is None else SPEC["fit_steps"] + 1):
                if step and optimizer is not None:
                    ix = torch.randint(len(train), (SPEC["batch_size"],), device="cuda")
                    loss = (codec.decode(ztrain[ix].float()) - train[ix].float()).square().mean()
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(codec.head.parameters(), 1.0)
                    optimizer.step()
                if step % SPEC["validation_every"] == 0:
                    with torch.no_grad():
                        deployed = copy.deepcopy(codec).bfloat16().eval()
                        error = (
                            (deployed.decode(zval).float() - validation.float())
                            .square()
                            .mean()
                            .item()
                        )
                        assert torch.isfinite(torch.tensor(error))
                        if error < best:
                            best, best_step = error, step
                            best_state = {
                                k: v.cpu().clone() for k, v in deployed.state_dict().items()
                            }
                        del deployed
            assert sum(x.numel() for x in best_state.values()) == parameter_elements(method, d, r)
            saved[method].append(best_state)
            rows.append(
                {
                    "layer": layer,
                    "method": method,
                    "rank": r,
                    "validation_mse": best,
                    "selected_step": best_step,
                    "parameter_bytes": sum(
                        x.numel() * x.element_size() for x in best_state.values()
                    ),
                }
            )
            print("FIT", rows[-1], flush=True)
        del train, validation, centered, vectors
    torch.save(saved, RUN / "codecs.pt")
    save_json(
        RUN / "fit.json",
        {
            "rows": rows,
            "codecs_sha256": sha(RUN / "codecs.pt"),
            "values_sha256": sha(RUN / "values.pt"),
            "protocol_sha256": sha(__file__.replace("fit.py", "protocol.json")),
        },
    )


if __name__ == "__main__":
    main()
