# Track 3 submission image: a fast, semantics-preserving re-implementation of the ABIDES baseline.
#
#   docker build --platform=linux/amd64 -t track3-fastsim:latest fastsim/
#
# `simulate` / `simulate-batch` are a native binary (csrc/native_main.cpp): scenario parsing,
# NumPy-identical seeding, the event engine and parquet writing, with no interpreter start-up. It
# writes parquet through the libarrow/libparquet shipped in the pyarrow 15.0.2 wheel (the library
# that wrote the reference traces), linked from site-packages. Anything the native path is not
# certain about is handed to the Python implementation (`python3 -m fastsim.simulate`), which is
# kept in the image with the baseline's run-time stack (Python 3.11, numpy 1.26.4, pyarrow 15.0.2).
# Nothing is fetched at run time (`--network none`).

# ---- build stage (the full image ships g++) --------------------------------------------------
FROM --platform=linux/amd64 python:3.11@sha256:27e044f7e01fea05c1760324d58fc5360a0767b9ef098e74ddaf8c70b8f46d26 AS build
WORKDIR /src
# An optional build secret `ca` (a CA bundle) is honoured for pip behind a TLS-intercepting proxy.
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --no-cache-dir setuptools==75.8.0 numpy==1.26.4 pyarrow==15.0.2
COPY setup.py ./
COPY csrc ./csrc
COPY fastsim ./fastsim
# Python extension (used by the fallback path) ...
RUN python setup.py -q build_ext --inplace && ls fastsim/_engine*.so
# ... and the native binary. No fast-math, no FMA contraction (floating-point order = NumPy's).
RUN PA=$(python -c "import pyarrow, os; print(os.path.dirname(pyarrow.__file__))") \
    && g++ -O3 -std=c++17 -ffp-contract=off -fno-fast-math -w -I"$PA/include" \
       csrc/native_main.cpp csrc/native_setup.cpp csrc/engine.cpp csrc/writer.cpp csrc/svml/svml_z0_log_d_la.s \
       -L"$PA" -l:libarrow.so.1500 -l:libparquet.so.1500 -Wl,-rpath,"$PA" \
       -static-libstdc++ -static-libgcc -o /src/fastsim-native

# ---- run-time stage ----------------------------------------------------------------------------
FROM --platform=linux/amd64 python:3.11-slim@sha256:6f31d6e9ba2b0a787a3f81c37b004155b87b9efa1b771182bd550c1615745be5

LABEL qfbench2.interface_version="2.0"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/fastsim

RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install numpy==1.26.4 pyarrow==15.0.2

COPY --from=build /src/fastsim /opt/fastsim/fastsim
COPY --from=build /src/fastsim-native /usr/local/bin/fastsim-native
RUN ln -s fastsim-native /usr/local/bin/simulate \
    && ln -s fastsim-native /usr/local/bin/simulate-batch \
    && rm -rf /opt/fastsim/fastsim/__pycache__ \
    && python -m compileall -q /opt/fastsim \
    && python -c "import fastsim._engine" \
    && simulate --help >/dev/null

WORKDIR /work
CMD ["simulate", "--help"]
