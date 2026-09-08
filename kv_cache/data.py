"""Select articles before any model evaluation; retain original human QA labels."""

import json
import random
import urllib.request

from common import RUN, SPEC, save_json, sha, tokenizer


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    assert not (RUN / "documents.json").exists(), "Refusing to change an existing split"
    tok = tokenizer()
    prompt = tok.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Answer the question using only the document. Return only the shortest answer span, with no explanation.\n\nDocument:\nDOC_MARKER\n\nQuestion: QUESTION_MARKER",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    start, tail = prompt.split("DOC_MARKER")
    ending = tail.split("QUESTION_MARKER")[1]
    sources, pools = {}, {}
    rng = random.Random(SPEC["seed"])
    for split in ["train", "dev"]:
        url = f"https://rajpurkar.github.io/SQuAD-explorer/dataset/{split}-v1.1.json"
        path = RUN / f"squad-{split}.json"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        sources[split] = {"url": url, "sha256": sha(path)}
        articles = json.loads(path.read_text())["data"]
        rng.shuffle(articles)
        selected = []
        for article in articles:
            text = "\n\n".join(p["context"] for p in article["paragraphs"])
            prefix = tok.encode(start + text, add_special_tokens=False)
            if len(prefix) < SPEC["prefix_tokens"]:
                continue
            prefix = prefix[: SPEC["prefix_tokens"]]
            visible = tok.decode(prefix)
            questions = []
            for para in article["paragraphs"]:
                # Preserve context grounding; later paragraphs outside the prefix are excluded.
                if para["context"] not in visible:
                    continue
                candidates = [
                    q for q in para["qas"] if all(a["text"] in visible for a in q["answers"])
                ]
                if candidates:
                    q = rng.choice(candidates)
                    questions.append(
                        {
                            "id": q["id"],
                            "question": q["question"],
                            "answers": [a["text"] for a in q["answers"]],
                            "suffix": tok.encode(
                                "\n\nQuestion: " + q["question"] + ending, add_special_tokens=False
                            ),
                        }
                    )
            if len(questions) >= 2:
                selected.append(
                    {
                        "title": article["title"],
                        "prefix": prefix,
                        "questions": [questions[0], questions[-1]],
                    }
                )
        pools[split] = selected
    ntrain, nval, ntest = (SPEC[f"{s}_documents"] for s in ["train", "validation", "test"])
    documents = {
        "train": pools["train"][:ntrain],
        "validation": pools["dev"][:nval],
        "test": pools["dev"][nval : nval + ntest],
    }
    assert [len(documents[s]) for s in documents] == [ntrain, nval, ntest]
    titles = [d["title"] for ds in documents.values() for d in ds]
    prefixes = [tuple(d["prefix"]) for ds in documents.values() for d in ds]
    qids = [q["id"] for ds in documents.values() for d in ds for q in d["questions"]]
    assert len(titles) == len(set(titles)) and len(prefixes) == len(set(prefixes))
    assert len(qids) == len(set(qids))
    save_json(RUN / "documents.json", documents)
    save_json(
        RUN / "data-manifest.json",
        {
            "sources": sources,
            "documents_sha256": sha(RUN / "documents.json"),
            "protocol_sha256": sha(__file__.replace("data.py", "protocol.json")),
            "titles": {s: [d["title"] for d in ds] for s, ds in documents.items()},
        },
    )
    print({s: len(ds) for s, ds in documents.items()}, flush=True)


if __name__ == "__main__":
    main()
