#!/usr/bin/env python3
"""Upload the converted artifacts to the CognitiveOS/vision HF repo.

Reads HF_TOKEN from .credentials/.hf_token (workspace root). Pushes:
  - model card README.md
  - .gitattributes (with *.gguf tracked as LFS)
  - per model: <model>.gguf, <model>.gguf-mapping.json,
               <model>.safetensors, <model>.safetensors.mapping.json
"""

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

WORKSPACE = Path("/workspace")
TOKEN_FILE = WORKSPACE / ".credentials" / ".hf_token"
ARTIFACTS = WORKSPACE / "packages" / "face-recognition" / "artifacts"
CARD_DIR = WORKSPACE / "packages" / "face-recognition" / ".hf-model-card"
REPO = "CognitiveOS/vision"

GITATTR_LINE = "*.gguf filter=lfs diff=lfs merge=lfs -text"

MODELS = [
    "tiny_face_detector_model",
    "face_landmark_68_model",
    "face_landmark_68_tiny_model",
    "face_recognition_model",
    "ssd_mobilenetv1_model",
    "age_gender_model",
    "face_expression_model",
]


def main():
    token = TOKEN_FILE.read_text().strip()
    api = HfApi(token=token)

    try:
        repo_info = api.model_info(REPO, token=token)
    except RepositoryNotFoundError:
        sys.exit(f"repo not found: {REPO}")

    existing = {f.rfilename for f in repo_info.siblings}

    # 1. README model card
    readme = CARD_DIR / "README.md"
    print(f"uploading {readme.name}")
    api.upload_file(
        repo_id=REPO, path_or_fileobj=readme.read_bytes(),
        path_in_repo="README.md", token=token,
    )

    # 2. .gitattributes (add gguf LFS rule, keep existing)
    try:
        cur = api.hf_hub_download(
            repo_id=REPO, filename=".gitattributes", token=token,
            cache_dir="/tmp/lab/hf-cache-gitattr",
        )
        cur = Path(cur).read_text()
    except Exception:
        cur = ""
    if GITATTR_LINE not in cur:
        cur = cur + GITATTR_LINE + "\n" if cur else GITATTR_LINE + "\n"
    print("uploading .gitattributes")
    api.upload_file(
        repo_id=REPO, path_or_fileobj=cur.encode(),
        path_in_repo=".gitattributes", token=token,
    )

    # 3. per-model artifacts
    for model in MODELS:
        model_dir = ARTIFACTS / model
        files = [
            f"{model}.gguf",
            f"{model}.gguf-mapping.json",
            f"{model}.safetensors",
            f"{model}.safetensors.mapping.json",
        ]
        for name in files:
            src = model_dir / name
            if not src.exists():
                print(f"  ! missing {name}")
                continue
            if name in existing:
                print(f"  = {name} (exists)")
                continue
            print(f"  + {name}")
            api.upload_file(
                repo_id=REPO, path_or_fileobj=src.read_bytes(),
                path_in_repo=name, token=token,
            )

    # 4. summary
    info = api.model_info(REPO, token=token)
    print(f"\n{REPO} now has {len(info.siblings)} files")
    for f in sorted(info.siblings, key=lambda x: x.rfilename):
        size = f.size or 0
        print(f"  {size:>12,}  {f.rfilename}")


if __name__ == "__main__":
    main()
