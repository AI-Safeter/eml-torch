"""Freeze disjoint operand-pair partitions before circuit selection."""

import hashlib
import json
import random

from common import MODEL_ID, OUT, REVISION, dump

rng = random.Random(20260907)
pools = {k: [] for k in ["train", "validation", "test", "carry", "range"]}
for a in range(10, 90):
    for b in range(a, 90):
        if a + b >= 100:
            continue
        if max(a, b) >= 70:
            key = "range"
        elif a % 10 + b % 10 >= 16:
            key = "carry"
        else:
            h = int(hashlib.sha256(f"{a},{b}".encode()).hexdigest()[:8], 16) % 10
            key = "train" if h < 6 else ("validation" if h < 8 else "test")
        pools[key].append((a, b))
# Change the larger operand, keeping both full problems in the same partition.
records = {}
for key, pool in pools.items():
    rows = []
    for small, large in pool:
        candidates = [
            large2
            for small2, large2 in pool
            if small2 == small and (large2 + small) // 10 != (large + small) // 10
        ]
        if not candidates:
            continue
        c = rng.choice(candidates)
        rows.append(
            {
                "a": large,
                "b": small,
                "c": c,
                "sum": large + small,
                "corrupt_sum": c + small,
                "carry": large % 10 + small % 10 >= 10,
                "reserved_carry": large % 10 + small % 10 >= 16,
            }
        )
    rng.shuffle(rows)
    cap = {"train": 512, "validation": 128, "test": 128, "carry": 96, "range": 96}[key]
    records[key] = rows[:cap]
    assert len(records[key]) >= 16, (key, len(records[key]))
# No unordered clean or corrupted problem appears in more than one partition.
sets = {
    k: {tuple(sorted((r[a], r["b"]))) for r in rows for a in ["a", "c"]}
    for k, rows in records.items()
}
for a in sets:
    for b in sets:
        if a != b:
            assert not sets[a] & sets[b], (a, b)
dump("problems.json", records)
dump(
    "protocol.json",
    {
        "model": MODEL_ID,
        "revision": REVISION,
        "seed": 20260907,
        "counts": {k: len(v) for k, v in records.items()},
        "arithmetic": "two-digit positive integer addition, two-digit results; no model weights trained",
        "component": "last prompt position MLP-output rank-one orthogonal contribution",
        "selection": "train full-MLP interchange patching; top three layers; first four training-output PCA directions per layer; validation selects direction by mean clean-answer-versus-corrupt-answer first-digit logit margin effect",
        "features": "first three PCs of selected MLP input, training statistics only; predict downstream scalar component coefficient",
        "fit": "EML depth2/3/4, seeds0/1/2, population1024, generations50, LBFGS200; select validation scalar MSE",
        "comparisons": [
            "linear",
            "quadratic",
            "cubic",
            "Fourier with linear terms",
            "small MLP",
            "mean",
            "shuffled coefficients",
            "matched unrelated direction",
            "original restoration",
        ],
        "intervention": "convex mixtures of clean/corrupt upstream MLP inputs at alpha 0,.25,.5,.75,1; train/validation pairs only for fitting intervention-augmented candidates; test pairs sealed",
        "tests": [
            "unseen operand pairs",
            "units-sum >=16 held out from fitting",
            "operand >=70 held out",
            "symbolic and code formats held out",
        ],
        "primary_metrics": [
            "full greedy answer exact match",
            "first-digit KL and agreement",
            "component normalized RMSE",
            "paired first-digit margin effect agreement and restoration",
        ],
        "success": "replacement <=1 percentage point exact-answer accuracy loss; intervention normalized RMSE <=.2 and correlation >=.9; no blanket novelty/speed claim",
        "pilot": "0.6B unreliable under tested prompt formats; 1.7B passed eight illustrative prompt checks; pilot pairs excluded below if present",
    },
)
# Pilot operands are excluded from every scored/training prompt, including corrupt prompts.
pilot = {
    tuple(sorted(p))
    for p in [(17, 26), (38, 45), (29, 29), (57, 36), (11, 22), (43, 15), (62, 19), (48, 37)]
}
records = {
    k: [
        r
        for r in rows
        if tuple(sorted((r["a"], r["b"]))) not in pilot
        and tuple(sorted((r["c"], r["b"]))) not in pilot
    ]
    for k, rows in records.items()
}
dump("problems.json", records)

p = json.loads((OUT / "protocol.json").read_text())
p["counts"] = {k: len(v) for k, v in records.items()}
dump("protocol.json", p)
print(p["counts"])
