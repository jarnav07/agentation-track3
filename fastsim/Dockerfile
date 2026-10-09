# Track 3 submission image: a fast, semantics-preserving re-implementation of the ABIDES baseline.
#
#   docker build --platform=linux/amd64 -t track3-fastsim:latest fastsim/
#
# `simulate` / `simulate-batch` are a native binary (csrc/native_main.cpp): scenario parsing,
# NumPy-identical seeding, the event engine and parquet writing, with no interpreter start-up and
# no shared libraries beyond libc / libm. Its parquet writer (csrc/pqfast.cpp) reproduces the
# bytes of the pyarrow 15.0.2 wheel's parquet-cpp (the library that wrote the reference traces),
# with that wheel's snappy version vendored (csrc/third_party/snappy). Anything the native path is not
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
# ... and the native binary, with profile-guided optimisation from a training run over pgo/*.json
# (public scenarios; see pgo/README.md). No fast-math, no FMA contraction: the floating-point
# sequence is NumPy's, so the profile changes code layout only, never output.
COPY pgo ./pgo
RUN CXXF="-O3 -std=c++17 -ffp-contract=off -fno-fast-math -w" \
    && SNAPPY="snappy snappy-sinksource snappy-stubs-internal" \
    && build() { out=$1; shift; mkdir -p obj; \
         for f in native_main native_setup engine writer pqfast; do g++ $CXXF "$@" -c csrc/$f.cpp -o obj/$f.o || return 1; done; \
         for f in $SNAPPY; do g++ -O2 -std=c++17 -w -DHAVE_CONFIG_H "$@" -c csrc/third_party/snappy/$f.cc -o obj/$f.o || return 1; done; \
         g++ "$@" obj/*.o csrc/svml/svml_z0_log_d_la.s -static-libstdc++ -static-libgcc -pthread -o $out; } \
    && build /tmp/fastsim-train -fprofile-generate -fprofile-update=atomic \
    && (for f in pgo/*.json; do /tmp/fastsim-train simulate --config $f --out /tmp/train/trace.parquet >/dev/null || exit 1; done \
        || (echo "PGO training failed; building without a profile"; rm -f obj/*.gcda)) \
    && rm -f obj/*.o && build /src/fastsim-native -fprofile-use -fprofile-correction -Wno-missing-profile \
    && ldd /src/fastsim-native

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
