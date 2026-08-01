# face-recognition CGP — reproducible model conversion pipeline.
#
# Starts from a fresh clone of vladmandic/face-api at a pinned commit,
# extracts the TF.js weight blobs, and produces safetensors + gguf artifacts
# plus mapping JSONs for every model, running the full validation suite.
#
# Build:
#   docker build -t face-recognition-convert .
#
# Extract artifacts:
#   id=$(docker create face-recognition-convert)
#   docker cp $id:/artifacts ./artifacts-out
#   docker rm $id
#
# The final stage is `scratch`; the artifacts live at /artifacts.

# Stage 1: extract TF.js weights (Node + pure-JS tfjs, no native bindings)
FROM node:22-alpine AS extract
ARG FACE_API_COMMIT=189226d63aabb48cb40776fd1c453ebc0fa722f1
RUN apk add --no-cache git
WORKDIR /work
RUN git clone --filter=blob:none https://github.com/vladmandic/face-api.git \
        upstream/face-api \
 && git -C upstream/face-api checkout ${FACE_API_COMMIT}
COPY scripts /work/scripts
RUN cd /work/scripts && npm install --silent @tensorflow/tfjs
RUN cd /work/scripts && node extract-tfjs-weights.js \
        --model-dir /work/upstream/face-api/model \
        --out /work/artifacts

# Stage 2: convert + validate (Python)
FROM python:3.12-slim AS convert
WORKDIR /work
COPY scripts /work/scripts
COPY --from=extract /work/artifacts /work/artifacts
COPY --from=extract /work/upstream/face-api/model /work/upstream/face-api/model
RUN pip install --no-cache-dir -r /work/scripts/requirements.txt
RUN python /work/scripts/verify_tfjs_bin.py \
        --artifacts /work/artifacts \
        --model-dir /work/upstream/face-api/model
RUN python /work/scripts/convert-to-safetensors.py --artifacts /work/artifacts
RUN python /work/scripts/convert-to-gguf.py --artifacts /work/artifacts
RUN python /work/scripts/read-gguf.py --all --artifacts /work/artifacts
RUN python /work/scripts/check_gguf_external.py --artifacts /work/artifacts
RUN python /work/scripts/read-gguf.py --all --artifacts /work/artifacts \
    > /artifacts/VALIDATION.log \
 && python /work/scripts/check_gguf_external.py --artifacts /work/artifacts \
    >> /artifacts/VALIDATION.log

# Stage 3: deliver only the artifacts
FROM scratch AS final
COPY --from=convert /work/artifacts /artifacts
