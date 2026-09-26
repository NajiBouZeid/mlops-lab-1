# Lab 3 - Answers

> **Matricule:** 231833
> **Full Name:** Naji Bou Zeid
> **Group:** MLOps grp 7
> **Date:** 26-9-2026

## Notes on My Setup

A few adjustments were needed to make the lab run on my machine (Windows + Docker Desktop, mlflow 3.16). Each change is also explained in a comment inside the `Dockerfile`.

- **`pyproject.toml`:** I added `[tool.uv] package = false`. `uv init` had set the project up as an installable package, so `uv sync` inside Docker failed looking for a module that isn't copied at that stage:
  ```
  => ERROR [builder 5/5] RUN uv sync --frozen --no-dev
  0.415    Building mlops-lab-1 @ file:///app
  0.432 error: Failed to build `mlops-lab-1 @ file:///app`
  0.432   cause: Expected a Python module at: src/mlops_lab_1/__init__.py
  ```
  The project is an application, not a library, so this setting is the correct one. I also added `--no-install-project` to the `uv sync` line in the Dockerfile.
- **`Dockerfile`:** I set `ENV UV_HTTP_TIMEOUT=300` in the builder stage because downloads timed out on my connection:
  ```
  Failed to download distribution due to network timeout. Try increasing UV_HTTP_TIMEOUT (current value: 30s).
  ```
  I also pinned the base image to a digest (`python:3.11-slim@sha256:da047cb8...`), for the reason explained in Question 4.
- **mlflow server:** I start it with `--host 0.0.0.0 --allowed-hosts "host.docker.internal:5000,127.0.0.1:5000,localhost:5000"`, and I run the container with a bind mount of my `mlruns/` folder (explained in Question 7).

## Question 1: Open the "Models" tab in the mlflow UI. What version number was your model given? What's the difference between a run's logged model artifact and a registered model?

I registered the model from my best Lab 2 run, `unequaled-mule-30` (run ID `a9af223f8a68490f9095255ebfe38297`), and it was given **version 1**:

```
PS ...\mlops-lab-1> uv run python register_model.py
Successfully registered model 'food11'.
2026/09/23 09:55:03 WARNING mlflow.tracking._model_registry.fluent: Run with id a9af223f8a68490f9095255ebfe38297 has no artifacts at artifact path 'model', registering model based on models:/m-057e722e1e3c4c0699c2cd4afbe474d5 instead
2026/09/23 09:55:03 INFO mlflow.store.model_registry.abstract_store: Waiting up to 300 secondsfor model version to finish creation. Model name: food11, version 1
Created version '1' of model 'food11'.
```

![food11 registered model, version 1](screenshots/lab3/lab3-q1-model-versions.png)

A **logged model artifact** is just the set of files a run produced (`MLmodel`, weights, `conda.yaml`, `requirements.txt`...). It belongs to that run and is identified by the run or by its model ID. The warning above shows this: in mlflow 3, the run's model is stored as its own logged model (`m-057e722e...`), and the registry version was created from it. A **registered model** is a named entry in the Model Registry (`food11`) with numbered versions. Each version points to one logged artifact and keeps a link back to the run that produced it (the version page shows `Source Run: unequaled-mule-30`). The registry gives the model its own name, version history, aliases and tags, independent of the experiment it came from.

## Question 2: What aliases replaced the old built-in stages in mlflow? Why version a model separately from the run that produced it, and why is an alias more flexible than a fixed stage name?

The fixed stages (`Staging`, `Production`, `Archived`) were replaced by **aliases**: free-form names chosen by the user, such as `champion` or `challenger`, each pointing to one model version. I assigned `champion` to version 1 with:

```python
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.MlflowClient()
client.set_registered_model_alias("food11", "champion", 1)
```

The version page shows the alias, and the old field as `Stage (deprecated): None`:

![Version 1 with the champion alias](screenshots/lab3/lab3-q2-champion-alias.png)

Versioning the model separately from the run separates *experimenting* from *deploying*. A run is a record of one training experiment. A registered version is a deployable unit that can be promoted, compared or rolled back, while still keeping its lineage to the run. Many runs may never be worth registering; only the chosen ones become versions.

An alias is more flexible than a fixed stage because:
- I choose the names, and a model can have several aliases at once (for example `champion` and `challenger`), instead of a fixed list of stages.
- An alias can be moved to another version at any time without changing version numbers.
- Serving code refers to the alias (`models:/food11@champion`), so promoting a new model means moving the alias, not editing code.

## Question 3: Why load the model through an mlflow model URI (`models:/food11@champion`) instead of pointing directly at the `.pth` file on disk? What would you have to change to serve a newer model version?

With the URI, the serving code doesn't need to know where the model files are or what format they are in. mlflow resolves `food11@champion` to a version, finds that version's artifact location, downloads it, and loads it with the right flavor, using the `MLmodel` metadata. A hardcoded `.pth` path would tie the code to one file on one machine and would require the code to rebuild the model architecture itself.

The resolution is visible in the mlflow server log when the API starts: it looks up the alias, then the model, then the download location of version 1:

```
INFO:     127.0.0.1:64929 - "GET /api/2.0/mlflow/registered-models/alias?name=food11&alias=champion HTTP/1.1" 200 OK
INFO:     127.0.0.1:64929 - "GET /api/2.0/mlflow/registered-models/get?name=food11 HTTP/1.1" 200 OK
INFO:     127.0.0.1:64929 - "GET /api/2.0/mlflow/model-versions/get-download-uri?name=food11&version=1 HTTP/1.1" 200 OK
```

My local test of the API worked end to end:

```
PS ...\mlops-lab-1> uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000
PS ...\mlops-lab-1> curl.exe -X POST -F "file=@data/food11_processed_mini/validation/Bread/0_0.jpg" http://127.0.0.1:8000/predict
{"category":"Fried food","confidence":0.9877711534500122}
```

This particular prediction is wrong (the image is Bread). That reflects the model's accuracy (about 0.70 `val_accuracy` after 5 epochs on the mini dataset), not a problem with the serving pipeline.

To serve a newer version, no code change and no image rebuild are needed. I would register the new model, which becomes version 2, move the alias with `client.set_registered_model_alias("food11", "champion", 2)`, and restart the service. The restart is needed because my `serve.py` loads the model once at startup.

## Question 4: Why copy `pyproject.toml`/`uv.lock` and run `uv sync` *before* copying the rest of the source code, instead of copying everything at once? What happens to the build cache when you only change a line in `serve.py`?

Docker caches each instruction as a layer, and a layer is rebuilt only if its inputs, or any layer before it, changed. Dependencies change rarely; source code changes often. Copying `pyproject.toml`/`uv.lock` and running `uv sync` first means the expensive dependency layer depends only on those two files. Changing `serve.py` then invalidates only the `COPY src/` layer at the end. If everything were copied at once before `uv sync`, any code change, even a comment, would invalidate the copy step and force `uv sync` to reinstall every dependency.

The full build, which had to run `uv sync`, took over 4.7 hours on my connection:

```
[+] Building 17136.1s (12/13)
 => CACHED [builder 3/5] WORKDIR /app                                                 0.0s
 => CACHED [builder 4/5] COPY pyproject.toml uv.lock ./                               0.0s
 => [builder 5/5] RUN uv sync --frozen --no-dev --no-install-project              11311.0s
 => [runtime 3/4] COPY --from=builder /app/.venv /app/.venv                          43.7s
 => [runtime 4/4] COPY src/ ./src/                                                    0.7s
```

To test the cache, I then added a comment line to `serve.py` and rebuilt:

```
PS ...\mlops-lab-1> docker build -t food11-api:latest .
[+] Building 66.5s (13/13) FINISHED                                   docker:desktop-linux
 => [internal] load build definition from Dockerfile                                  0.0s
 => => transferring dockerfile: 973B                                                  0.0s
 => [internal] load metadata for docker.io/library/python:3.11-slim@sha256:da047cb8f  0.1s
 => [internal] load .dockerignore                                                     0.0s
 => => transferring context: 176B                                                     0.0s
 => [internal] load build context                                                     0.0s
 => => transferring context: 2.51kB                                                   0.0s
 => [builder 1/5] FROM docker.io/library/python:3.11-slim@sha256:da047cb8f9d1d98e5c0  0.0s
 => CACHED [runtime 2/4] WORKDIR /app                                                 0.0s
 => CACHED [builder 2/5] RUN pip install --no-cache-dir uv                            0.0s
 => CACHED [builder 3/5] WORKDIR /app                                                 0.0s
 => CACHED [builder 4/5] COPY pyproject.toml uv.lock ./                               0.0s
 => CACHED [builder 5/5] RUN uv sync --frozen --no-dev --no-install-project           0.0s
 => CACHED [runtime 3/4] COPY --from=builder /app/.venv /app/.venv                    0.0s
 => [runtime 4/4] COPY src/ ./src/                                                    0.0s
 => exporting to image                                                               65.9s
```

Every dependency step was `CACHED`; only `COPY src/` ran. The build took 66.5s, almost all of it unpacking the image, instead of 17,136s.

I also saw the other side of this rule. Between two builds, the official `python:3.11-slim` tag was updated upstream, so Docker resolved it to a new digest (`sha256:e41613d4...` instead of `sha256:da047cb8...`). Since every layer sits on top of the base image, all my cached layers were invalidated, even `pip install uv`:

```
 => CACHED [builder 1/5] FROM docker.io/library/python:3.11-slim@sha256:e41613d42d48  0.1s
 => ERROR [builder 2/5] RUN pip install --no-cache-dir uv                           290.0s
```

I pinned the base image to the digest I had already built with (`python:3.11-slim@sha256:da047cb8...`) to get stable caching.

## Question 5: What's the size difference between a naive single-stage image and your multi-stage one? Use `docker history <image>` to see which layers are the biggest.

I built a naive single-stage version (`Dockerfile.naive`: one stage, `COPY . .`, then `uv sync`) and compared:

```
PS ...\mlops-lab-1> docker images food11-api
IMAGE               ID             DISK USAGE   CONTENT SIZE   EXTRA
food11-api:latest   7594c8ee2667       9.59GB         3.25GB
food11-api:naive    54d83a627176       9.73GB         3.28GB
```

The multi-stage image is only about 30MB smaller in content size (3.25GB vs 3.28GB). I compare CONTENT SIZE because the DISK USAGE column also counts the unpacked copy Docker keeps for running containers. `docker history` of the multi-stage image shows why the gap is small:

```
PS ...\mlops-lab-1> docker history food11-api:latest
IMAGE          CREATED         CREATED BY                                      SIZE      COMMENT
7594c8ee2667   4 minutes ago   CMD ["uvicorn" "src.food11.serve:app" "--hos…   0B        buildkit.dockerfile.v0
<missing>      4 minutes ago   EXPOSE [8000/tcp]                               0B        buildkit.dockerfile.v0
<missing>      4 minutes ago   ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000   0B        buildkit.dockerfile.v0
<missing>      4 minutes ago   ENV PATH=/app/.venv/bin:/usr/local/bin:/usr/…   0B        buildkit.dockerfile.v0
<missing>      4 minutes ago   COPY src/ ./src/ # buildkit                     49.2kB    buildkit.dockerfile.v0
<missing>      22 hours ago    COPY /app/.venv /app/.venv # buildkit           6.2GB     buildkit.dockerfile.v0
<missing>      41 hours ago    WORKDIR /app                                    8.19kB    buildkit.dockerfile.v0
<missing>      7 days ago      CMD ["python3"]                                 0B        buildkit.dockerfile.v0
<missing>      7 days ago      RUN /bin/sh -c set -eux;  for src in idle3 p…   16.4kB    buildkit.dockerfile.v0
<missing>      7 days ago      RUN /bin/sh -c set -eux;   savedAptMark="$(a…   48.8MB    buildkit.dockerfile.v0
<missing>      7 days ago      ENV PYTHON_SHA256=91bcdebfdde239a003ae93738a…   0B        buildkit.dockerfile.v0
<missing>      7 days ago      ENV PYTHON_VERSION=3.11.16                      0B        buildkit.dockerfile.v0
<missing>      7 days ago      ENV GPG_KEY=A035C8C19219BA821ECEA86B64E628F8…   0B        buildkit.dockerfile.v0
<missing>      7 days ago      RUN /bin/sh -c set -eux;  apt-get update;  a…   4.95MB    buildkit.dockerfile.v0
<missing>      7 days ago      ENV LANG=C.UTF-8                                0B        buildkit.dockerfile.v0
<missing>      7 days ago      ENV PATH=/usr/local/bin:/usr/local/sbin:/usr…   0B        buildkit.dockerfile.v0
<missing>      8 days ago      # debian.sh --arch 'amd64' out/ 'trixie' '@1…   87.6MB    debuerreotype 0.17
```

The `.venv` layer (6.2GB uncompressed) is almost the entire image; the base Debian and Python layers are about 140MB combined, and my code is 49kB. Multi-stage only removes build tooling from the final image, here `uv` and pip's leftovers, which is tens of MB. The virtual environment has to be in the final image either way, and it is dominated by `torch` with its CUDA libraries. The build log shows the largest downloads:

```
Downloading torch (528.9MiB)
Downloading nvidia-cudnn-cu13 (527.5MiB)
Downloading nvidia-cublas (403.5MiB)
```

With dependencies this large, the saving from multi-stage is real but small in proportion. The bigger reduction for this image would be installing CPU-only PyTorch, since the container has no GPU access.

## Question 6: What happens to build speed and image size if you forget the `.dockerignore`? Which of the excluded folders would actually break the build if they were sent to the Docker daemon?

I hit this by accident: my `.dockerignore` was saved empty at one point. Building the naive image then sent my entire project folder to the Docker daemon:

```
PS ...\mlops-lab-1> docker build -f Dockerfile.naive -t food11-api:naive .
 => [internal] load build context                                                   255.4s
 => => transferring context: 3.89GB                                                 254.9s
```

The transfer alone took over 4 minutes, and the build crashed Docker Desktop:

![Docker Desktop crash during the build without .dockerignore](screenshots/lab3/lab3-q6-docker-crash.png)

After restoring the `.dockerignore`, the same naive build sent almost nothing:

```
PS ...\mlops-lab-1> docker build -f Dockerfile.naive -t food11-api:naive .
 => [internal] load .dockerignore                                                     0.0s
 => => transferring context: 176B                                                     0.0s
 => [internal] load build context                                                     0.1s
 => => transferring context: 756.70kB                                                 0.1s
```

For **image size**, the effect depends on the Dockerfile. In the naive one, `COPY . .` would copy everything sent into the image: the datasets in `data/` and `local_data/`, `mlruns/`, `mlflow.db`, `.git/`. In my multi-stage one, only specific files are copied (`pyproject.toml`, `uv.lock`, `src/`). When I renamed the `.dockerignore` away and built the multi-stage image, only 420B of context were transferred, so its size is not affected:

```
PS ...\mlops-lab-1> Rename-Item .dockerignore .dockerignore.bak
PS ...\mlops-lab-1> docker build -t food11-api:test-no-ignore .
 => [internal] load .dockerignore                                                     0.0s
 => => transferring context: 2B                                                       0.0s
 => [internal] load build context                                                     0.0s
 => => transferring context: 420B                                                     0.0s
```

The folder that would actually **break** the build is `.venv/`. It is my host's virtual environment, created on Windows, with Windows executables and paths. In the naive Dockerfile, `COPY . .` would place it at `/app/.venv`, exactly where `uv sync` builds the Linux environment, so the container would start from an environment built for the wrong platform. The data folders don't break correctness, but in my case their size is what crashed Docker Desktop. `mlflow.db`, `mlruns/` and `.git/` add size and can expose private data, but don't break the build.

## Question 7: Why can't the container simply use `127.0.0.1:5000` to reach the mlflow server on your host? What does `host.docker.internal` resolve to?

A container has its own network namespace. Inside it, `127.0.0.1` is the container itself, and nothing listens on port 5000 there, so the connection is refused. `host.docker.internal` is a special hostname that Docker Desktop's DNS resolves to an internal IP address of the host machine, reachable from inside the container. I ran the container with:

```
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest
```

Reaching the host was not enough on its own. I hit two more issues.

**1. mlflow's security middleware** rejected the request, because mlflow 3 only accepts localhost `Host` headers by default. The container failed at startup:

```
mlflow.exceptions.MlflowException: API request to endpoint /api/2.0/mlflow/registered-models/alias failed with error code 403 != 200. Response body: 'Invalid Host header - possible DNS rebinding attack detected'

ERROR:    Application startup failed. Exiting.
```

and the mlflow server log showed why:

```
WARNING mlflow.server.fastapi_security: Rejected request with invalid Host header: host.docker.internal:5000
INFO:     127.0.0.1:15807 - "GET /api/2.0/mlflow/registered-models/alias?name=food11&alias=champion HTTP/1.1" 403 Forbidden
```

I restarted the server listening on all interfaces and allowing that host:

```
PS ...\mlops-lab-1> uv run mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --allowed-hosts "host.docker.internal:5000,127.0.0.1:5000,localhost:5000"
[MLflow] Security middleware enabled. Allowed hosts: host.docker.internal:5000, 127.0.0.1:5000, localhost:5000.
```

**2. Local artifact store:** the registry calls then succeeded (`200 OK`), but loading failed:

```
mlflow.exceptions.MlflowException: No such artifact: ''

ERROR:    Application startup failed. Exiting.
```

Because my artifact store is a local folder, the server returns a Windows file path (`file:///C:/Users/eabou/.../mlruns/...`), and the container tries to read that path from its own filesystem, where it doesn't exist. I fixed it by bind-mounting my `mlruns/` folder into the container at that same path:

```
docker run -p 8000:8000 `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
  --mount type=bind,source="C:\Users\eabou\Desktop\USJ 3eme annee\mlops\labs\mlops-lab-1\mlruns",target="/C:/Users/eabou/Desktop/USJ 3eme annee/mlops/labs/mlops-lab-1/mlruns" `
  food11-api:latest
```

After that, the containerized API returned the same prediction as the local one:

```
PS ...\mlops-lab-1> curl.exe -X POST -F "file=@data/food11_processed_mini/validation/Bread/0_0.jpg" http://127.0.0.1:8000/predict
{"category":"Fried food","confidence":0.9877711534500122}
```

## Question 8: Stop the container and start a new one from the same image. Does the model still load correctly without you rebuilding? What does that tell you about what's baked into the image versus fetched at runtime?

Yes. After shutting everything down (including a full restart of my laptop), I started the mlflow server again and a new container from the same image, without rebuilding. The model loaded and gave the same prediction:

```
PS ...\mlops-lab-1> curl.exe -X POST -F "file=@data/food11_processed_mini/validation/Bread/0_0.jpg" http://127.0.0.1:8000/predict
{"category":"Fried food","confidence":0.9877711534500122}
```

This shows that the image contains the **code and its environment** (Python, the dependencies in `.venv`, `serve.py`), but **not the model**. The model is fetched at runtime, at startup, from the registry through `models:/food11@champion`. A new container always loads whatever version `champion` points to at that moment, so the model can be updated without rebuilding the image. The trade-off is that every container depends on the tracking server and the artifact store being reachable when it starts: when my mlflow server wasn't running, the API failed at startup:

```
mlflow.exceptions.MlflowException: API request to http://127.0.0.1:5000/api/2.0/mlflow/registered-models/alias failed with exception HTTPConnectionPool(host='127.0.0.1', port=5000): Max retries exceeded ... No connection could be made because the target machine actively refused it

ERROR:    Application startup failed. Exiting.
```

## Question 9: The Dockerfile and image are versioned differently — one lives in git, the other doesn't (yet). What's still missing before another machine (like a CI runner or a Kubernetes cluster) could reliably pull and run the exact image you just built?

- **A container registry.** The image exists only in my local Docker. It has to be pushed to a registry (Docker Hub, GitHub Container Registry, AWS ECR...) so other machines can pull it.
- **An immutable version tag.** `latest` is overwritten on every build: in this lab alone, `food11-api:latest` pointed to three different images (`332910dc1135`, `366bff9d1f86`, then `7594c8ee2667`). The image should be tagged with something unique, such as the git commit SHA, and deployments should reference that tag or the image digest, so "the exact image" is identifiable and ties back to the commit that produced it.
- **Automated builds.** The image should be built by a CI pipeline from the committed Dockerfile, not by hand on my laptop, so that a given commit always produces the image that gets deployed. Pinning the base image digest is part of this: I saw the `python:3.11-slim` tag change between two of my own builds.
- **Shared runtime dependencies.** The container doesn't contain the model; it needs a tracking server and an artifact store at startup. My current setup relies on an mlflow server on my laptop, a local artifact folder, and a bind mount of a Windows path, none of which exist on a CI runner or a Kubernetes cluster. It would need a tracking server reachable from the cluster, a shared artifact store (such as S3 or MinIO), and the tracking URI and credentials passed as environment variables and secrets.
