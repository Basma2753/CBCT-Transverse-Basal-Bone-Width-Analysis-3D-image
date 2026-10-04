# CPU measurement/UI image. CUDA inference is a separately configured environment.
# Official Python multi-platform image digest resolved on 2026-10-04.
FROM python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CBCT_WORK_DIR=/work/.cbct-work

WORKDIR /opt/cbct
COPY pyproject.toml README.md requirements-lock.txt ./
COPY src ./src
COPY app.py ./app.py
RUN python -m pip install setuptools==80.9.0 wheel==0.45.1 \
    && python -m pip install --no-build-isolation -c requirements-lock.txt ".[ui]" \
    && python -m pip check \
    && groupadd --gid 10001 cbct \
    && useradd --uid 10001 --gid cbct --create-home cbct \
    && mkdir -p /work \
    && chown cbct:cbct /work
USER cbct
WORKDIR /work
ENTRYPOINT ["cbct-width"]
CMD ["--help"]
