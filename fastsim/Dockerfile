# Track 3 submission image: fast, semantics-preserving re-implementation of the ABIDES baseline.
#
#   docker build --platform=linux/amd64 -t track3-fastsim:latest fastsim/
#
# Same Python / NumPy / pyarrow as the baseline image (3.11, numpy 1.26.4, pyarrow 15.0.2): the
# seeded NumPy streams and the parquet writer (its version is written into the file footer) must be
# the ones that produced the reference traces. pandas, scipy and ABIDES are not needed at run time.
FROM --platform=linux/amd64 python:3.11-slim

LABEL qfbench2.interface_version="2.0"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/fastsim

# An optional build secret `ca` (a CA bundle) is honoured for pip behind a TLS-intercepting proxy;
# without it the install is a plain `pip install`.
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install numpy==1.26.4 pyarrow==15.0.2

COPY fastsim /opt/fastsim/fastsim
COPY bin/simulate bin/simulate-batch /usr/local/bin/
RUN chmod +x /usr/local/bin/simulate /usr/local/bin/simulate-batch \
    && python -m compileall -q /opt/fastsim

WORKDIR /work
CMD ["simulate", "--help"]
