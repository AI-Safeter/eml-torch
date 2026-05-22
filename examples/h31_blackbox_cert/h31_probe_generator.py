#!/usr/bin/env python3
"""H31 probe generator — 5 circuit classes × ~50 prompts.

Black-box safe: each prompt has a deterministic ground-truth target token
derivable from the prompt string alone, NOT from any model internals.

Output: outputs/h31_blackbox_cert/probes.jsonl
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "probes.jsonl"

SEED = 20260522
random.seed(SEED)


# ---------------------------------------------------------------------------
# Induction (reusing H19c pools)
# ---------------------------------------------------------------------------

INDUCTION_POOLS = {
    "celestial": [
        " moon",
        " book",
        " tree",
        " star",
        " cloud",
        " rain",
        " sun",
        " wave",
        " fire",
        " wind",
    ],
    "animals": [
        " cat",
        " dog",
        " bird",
        " fish",
        " bear",
        " wolf",
        " lion",
        " deer",
        " seal",
        " hawk",
    ],
    "fruits": [
        " apple",
        " grape",
        " mango",
        " peach",
        " plum",
        " berry",
        " melon",
        " lemon",
        " olive",
        " kiwi",
    ],
    "tech": [
        " code",
        " data",
        " file",
        " disk",
        " port",
        " host",
        " link",
        " path",
        " core",
        " bit",
    ],
    "body": [
        " hand",
        " foot",
        " head",
        " arm",
        " leg",
        " eye",
        " ear",
        " back",
        " neck",
        " hip",
    ],
    "colors": [
        " red",
        " blue",
        " green",
        " gold",
        " black",
        " white",
        " brown",
        " pink",
        " gray",
        " orange",
    ],
    "kitchen": [
        " cup",
        " plate",
        " bowl",
        " fork",
        " spoon",
        " knife",
        " pan",
        " pot",
        " mug",
        " jar",
    ],
    "metals": [
        " iron",
        " gold",
        " silver",
        " copper",
        " steel",
        " brass",
        " tin",
        " lead",
        " zinc",
        " nickel",
    ],
}


def gen_induction(n_per_pool: int = 7) -> list[dict]:
    """[tokens × 2 + tokens[0]] → target = tokens[1]. Vary lengths."""
    out = []
    rng = random.Random(SEED + 1)
    lengths = [8, 9, 10, 10, 10, 11, 12]  # length variation
    for pool_name, pool in INDUCTION_POOLS.items():
        for length in lengths[:n_per_pool]:
            toks = list(pool[:length])
            rng.shuffle(toks)
            prompt_tokens = toks + toks + [toks[0]]
            prompt = "".join(prompt_tokens).lstrip()
            out.append(
                {
                    "circuit": "induction",
                    "subclass": pool_name,
                    "prompt": prompt,
                    "target_token_str": toks[1],
                    "T": len(prompt_tokens),
                    "L": length,  # induction lag = pool size
                    "n_repeats": 2,
                    "first_prior_occurrence": length,  # index in token list
                    "last_q": 2 * length,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Copy one-shot (H21 pattern, vary cue word and content)
# ---------------------------------------------------------------------------

COPY_WORDS = [
    " apple",
    " mountain",
    " keyboard",
    " ocean",
    " python",
    " hammer",
    " velvet",
    " marble",
    " thunder",
    " silver",
    " forest",
    " crystal",
    " bronze",
    " harbor",
    " willow",
    " coffee",
    " window",
    " summer",
    " diamond",
    " bridge",
]
COPY_CUES = [" again", " repeat", " once more", " twice"]


def gen_copy(n: int = 50) -> list[dict]:
    out = []
    rng = random.Random(SEED + 2)
    for _ in range(n):
        w = rng.choice(COPY_WORDS)
        cue = rng.choice(COPY_CUES)
        prompt = f"Say{w}.{cue}:"
        out.append(
            {
                "circuit": "copy_oneshot",
                "subclass": "say_again",
                "prompt": prompt,
                "target_token_str": w,
                "T": len(prompt.split()),
                "L": 0,
                "n_repeats": 1,
                "cue": cue,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Factual recall (capitals + languages, H22e flavor)
# ---------------------------------------------------------------------------

# (country, capital, language). All capitals and languages are single-token
# in Qwen/Gemma tokenizers (verified by spot-check; verified at runtime).
CAPITALS = [
    ("France", " Paris", " French"),
    ("Germany", " Berlin", " German"),
    ("Japan", " Tokyo", " Japanese"),
    ("Italy", " Rome", " Italian"),
    ("Spain", " Madrid", " Spanish"),
    ("Russia", " Moscow", " Russian"),
    ("China", " Beijing", " Chinese"),
    ("Egypt", " Cairo", " Arabic"),
    ("Greece", " Athens", " Greek"),
    ("Turkey", " Ankara", " Turkish"),
    ("Poland", " Warsaw", " Polish"),
    ("Sweden", " Stockholm", " Swedish"),
    ("Norway", " Oslo", " Norwegian"),
    ("Finland", " Helsinki", " Finnish"),
    ("Brazil", " Brasilia", " Portuguese"),
    ("Mexico", " Mexico", " Spanish"),
    ("Canada", " Ottawa", " English"),
    ("Australia", " Canberra", " English"),
    ("India", " Delhi", " Hindi"),
    ("Vietnam", " Hanoi", " Vietnamese"),
    ("Thailand", " Bangkok", " Thai"),
    ("Indonesia", " Jakarta", " Indonesian"),
    ("Korea", " Seoul", " Korean"),
    ("Argentina", " Buenos", " Spanish"),
    ("Chile", " Santiago", " Spanish"),
]


def gen_factual(n: int = 50) -> list[dict]:
    out = []
    rng = random.Random(SEED + 3)
    # Two templates per pair → ~50
    for country, capital, language in CAPITALS:
        # capital template
        prompt_c = f"The capital of {country} is"
        out.append(
            {
                "circuit": "factual",
                "subclass": "capital_of",
                "prompt": prompt_c,
                "target_token_str": capital,
                "T": len(prompt_c.split()),
                "L": 0,
                "n_repeats": 0,
                "country": country,
            }
        )
        # language template
        prompt_l = f"The language spoken in {country} is"
        out.append(
            {
                "circuit": "factual",
                "subclass": "language_of",
                "prompt": prompt_l,
                "target_token_str": language,
                "T": len(prompt_l.split()),
                "L": 0,
                "n_repeats": 0,
                "country": country,
            }
        )
    rng.shuffle(out)
    return out[:n]


# ---------------------------------------------------------------------------
# IOI (Wang 2022 indirect-object identification)
# ---------------------------------------------------------------------------

NAMES = [
    " John",
    " Mary",
    " Alice",
    " Bob",
    " Carol",
    " David",
    " Emma",
    " Frank",
    " Grace",
    " Henry",
    " Iris",
    " Jack",
    " Kate",
    " Liam",
    " Nora",
    " Oscar",
]
OBJECTS = [
    " milk",
    " book",
    " ball",
    " key",
    " ring",
    " coin",
    " gift",
    " card",
    " note",
    " pen",
]


def gen_ioi(n: int = 50) -> list[dict]:
    out = []
    rng = random.Random(SEED + 4)
    for _ in range(n):
        a, b = rng.sample(NAMES, 2)
        obj = rng.choice(OBJECTS)
        # Wang 2022 ABBA template: A and B → B gave the obj to → A
        prompt = f"When{a} and{b} went to the store,{b} gave" f" the{obj} to"
        out.append(
            {
                "circuit": "ioi",
                "subclass": "wang_abba",
                "prompt": prompt,
                "target_token_str": a,
                "T": len(prompt.split()),
                "L": 0,
                "n_repeats": 0,
                "names": [a, b],
                "object": obj,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Syntactic (subject-verb agreement, gender pronoun resolution)
# ---------------------------------------------------------------------------

SUBJECTS_SING = [
    "cat",
    "dog",
    "bird",
    "child",
    "teacher",
    "doctor",
    "player",
    "writer",
    "student",
    "driver",
]
SUBJECTS_PLUR = [
    "cats",
    "dogs",
    "birds",
    "children",
    "teachers",
    "doctors",
    "players",
    "writers",
    "students",
    "drivers",
]


def gen_syntactic(n: int = 50) -> list[dict]:
    out = []
    rng = random.Random(SEED + 5)
    # Half subject-verb agreement
    for _ in range(n // 2):
        if rng.random() < 0.5:
            subj = rng.choice(SUBJECTS_SING)
            prompt = f"The {subj} is in the garden, and it"
            target = " is"
            subclass = "sva_sing"
        else:
            subj = rng.choice(SUBJECTS_PLUR)
            prompt = f"The {subj} are in the garden, and they"
            target = " are"
            subclass = "sva_plur"
        out.append(
            {
                "circuit": "syntactic",
                "subclass": subclass,
                "prompt": prompt,
                "target_token_str": target,
                "T": len(prompt.split()),
                "L": 0,
                "n_repeats": 0,
            }
        )
    # Half pronoun gender resolution
    male_names = [" John", " Bob", " David", " Frank", " Henry", " Jack"]
    female_names = [" Mary", " Alice", " Carol", " Emma", " Grace", " Iris"]
    for _ in range(n - n // 2):
        if rng.random() < 0.5:
            nm = rng.choice(male_names)
            prompt = f"After{nm} finished work,"
            target = " he"
            subclass = "pronoun_male"
        else:
            nm = rng.choice(female_names)
            prompt = f"After{nm} finished work,"
            target = " she"
            subclass = "pronoun_female"
        out.append(
            {
                "circuit": "syntactic",
                "subclass": subclass,
                "prompt": prompt,
                "target_token_str": target,
                "T": len(prompt.split()),
                "L": 0,
                "n_repeats": 0,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    probes = []
    probes.extend(gen_induction())  # ~56
    probes.extend(gen_copy())  # 50
    probes.extend(gen_factual())  # 50
    probes.extend(gen_ioi())  # 50
    probes.extend(gen_syntactic())  # 50

    # Stable probe IDs
    for i, p in enumerate(probes):
        p["probe_id"] = f"H31-{i:04d}"

    with OUT_PATH.open("w") as f:
        for p in probes:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    from collections import Counter

    by_circuit = Counter(p["circuit"] for p in probes)
    print(f"Wrote {len(probes)} probes → {OUT_PATH}")
    for c, n in sorted(by_circuit.items()):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
