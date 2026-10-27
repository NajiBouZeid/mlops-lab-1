FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.16.0

RUN mkdir -p /mlflow-data

EXPOSE 5000

# --artifacts-destination: the server stores artifacts on the volume and serves them over HTTP,
# so other containers download model files instead of reading a path they don't have.
# --allowed-hosts: mlflow 3 rejects Host headers it doesn't know, including "mlflow:5000".
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--artifacts-destination", "/mlflow-data/mlruns", \
     "--allowed-hosts", "mlflow:5000,localhost:5000,127.0.0.1:5000"]