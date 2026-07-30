FROM python:3.13-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends cargo rustc \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir maturin==1.14.1
COPY vendor/flopsearch /src/flopsearch
RUN cd /src/flopsearch/flop_python && maturin build --release --out /wheels

FROM python:3.13-slim
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl pandas==2.2.3
LABEL org.opencontainers.image.licenses="MPL-2.0"
