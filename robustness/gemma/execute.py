"""Run an unchanged scalar stage with the explicit Gemma adapter installed."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma.adapter import install

if __name__ == "__main__":
    script, *args = sys.argv[1:]
    install()
    sys.argv = [script, *args]
    module = importlib.import_module(
        "gemma.sparse_neurons" if script == "sparse_neurons" else script
    )
    module.main()
