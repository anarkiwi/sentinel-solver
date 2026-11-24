FROM python:3.15-slim AS deps
WORKDIR /app
COPY requirements.txt requirements-gpu.txt ./
RUN grep -v '^numpy' requirements.txt > base.txt \
    && pip install --no-cache-dir -r base.txt -r requirements-gpu.txt \
        black pylint pytest pytest-xdist

FROM deps AS test
COPY . .
ENV NUMBA_ENABLE_CUDASIM=1 SENTINEL_BACKEND=cpu \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
CMD ["pytest", "-n", "auto", "sentinel/tests/test_kern.py"]
