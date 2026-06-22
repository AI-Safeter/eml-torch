# emltorch: reproducible image for symbolic regression with the EML operator.
#
# Build:  docker build -t emltorch .
# REPL:   docker run --rm -it emltorch            # python, with emltorch importable
# Script: docker run --rm -v "$PWD:/work" -w /work emltorch python my_fit.py
#
# This is a CPU image (portable, ~1 GB). For GPU, replace the base image with a
# CUDA PyTorch image and drop the CPU torch install below, e.g.:
#   FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
#   ... (skip the "pip install torch --index-url .../cpu" line) ...
# then run with `docker run --gpus all ...`.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/emltorch

# CPU build of PyTorch first, so the [smt]/core install below does not pull the
# multi-GB CUDA wheels. z3 and cvc5 ship as manylinux wheels (the [smt] extra),
# so no system solver packages are needed.
RUN pip install --upgrade pip \
 && pip install "torch>=2.3" --index-url https://download.pytorch.org/whl/cpu

# Install emltorch with the SMT extra (z3 + cvc5) from the repo source.
COPY pyproject.toml README.md LICENSE ./
COPY emltorch ./emltorch
RUN pip install ".[smt]"

# Fail the build fast if the install is broken: a tiny end-to-end fit.
RUN python -c "import numpy as np, emltorch as eml; \
x = np.random.RandomState(0).uniform(-1, 1, (200, 2)); \
r = eml.fit(x, np.exp(x[:, 0] - x[:, 1]), depth=2, population=256, generations=8); \
print('emltorch', eml.__version__, 'ok ->', r.expression)"

CMD ["python"]
