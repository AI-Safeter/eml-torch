"""Build the standalone explorer from frozen, measured demo exports."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    paths = list(ROOT.glob("*/demo-*.json")) + list(ROOT.glob("replication-0.6b/*/demo-*.json"))
    data = [json.loads(path.read_text()) for path in paths]
    data.sort(
        key=lambda d: (
            0 if "1.7B" in d["model"] else 1,
            d["operation"],
            0 if d["mode"] == "primary" else 1,
        )
    )
    assert data
    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    html = (ROOT / "explorer.template.html").read_text().replace("_PAYLOAD_", payload)
    (ROOT / "explorer.html").write_text(html)
    print("EXPLORER BUILT", len(data), "datasets", len(html), "bytes")


if __name__ == "__main__":
    main()
