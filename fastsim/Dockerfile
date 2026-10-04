# Track 3 submission image: a fast, semantics-preserving re-implementation of the ABIDES baseline.
#
#   docker build --platform=linux/amd64 -t track3-fastsim:latest fastsim/
#
# Run-time stack = the baseline image's (Python 3.11, numpy 1.26.4, pyarrow 15.0.2): NumPy seeds
# the RNG streams before the loop and pyarrow writes the parquet (its version is in the footer), so
# both must be the versions that produced the reference traces. pandas, scipy and ABIDES are not
# needed at run time. Nothing is fetched at run time (`--network none`).

# ---- build stage: compile the engine extension (the full image ships g++) -------------------
FROM --platform=linux/amd64 python:3.11@sha256:27e044f7e01fea05c1760324d58fc5360a0767b9ef098e74ddaf8c70b8f46d26 AS build
WORKDIR /src
# An optional build secret `ca` (a CA bundle) is honoured for pip behind a TLS-intercepting proxy.
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --no-cache-dir setuptools==75.8.0
COPY setup.py ./
COPY csrc ./csrc
COPY fastsim ./fastsim
RUN python setup.py -q build_ext --inplace && ls fastsim/_engine*.so

# ---- run-time stage --------------------------------------------------------------------------
FROM --platform=linux/amd64 python:3.11-slim@sha256:6f31d6e9ba2b0a787a3f81c37b004155b87b9efa1b771182bd550c1615745be5

LABEL qfbench2.interface_version="2.0"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/fastsim

RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install numpy==1.26.4 pyarrow==15.0.2

COPY --from=build /src/fastsim /opt/fastsim/fastsim
COPY bin/simulate bin/simulate-batch /usr/local/bin/
RUN chmod +x /usr/local/bin/simulate /usr/local/bin/simulate-batch \
    && rm -rf /opt/fastsim/fastsim/__pycache__ \
    && python -m compileall -q /opt/fastsim \
    && python -c "import fastsim._engine"

WORKDIR /work
CMD ["simulate", "--help"]
