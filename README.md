# face-recognition

A **Cognitive Patch** (`.cgp`) for **CognitiveOS** — an independent operating
system where the AI is the OS. See [cognitive-os.org](https://cognitive-os.org).

This package converts the pre-trained weight blobs of the
[vladmandic/face-api](https://github.com/vladmandic/face-api) models into
portable **safetensors** and **gguf** artifacts, with a fully reproducible
extraction / export / verification pipeline. The input weights are the
official TF.js blobs checked into the face-api repository at commit
`189226d63aabb48cb40776fd1c453ebc0fa722f1` (`master` as of the conversion).

> There is **no `model.json` topology** in the face-api repo — the network
> architectures live in TypeScript under `src/`. This pipeline therefore
> converts **weights only** (name → tensor), preserving the original TF.js
> weight names verbatim for traceability. Rebuilding the networks in
> PyTorch/Keras from the TS sources and wiring these tensors in is the
> documented follow-up (see [Architecture reconstruction](#architecture-reconstruction)).

## Package architecture

This patch is one of two that share the converted weights:

| Package | Role |
|---------|------|
| **`face-recognition`** (this repo) | Installs the pretrained weights (declared as a remote Hugging Face model) and exposes face **detection / recognition / enrollment** tools via an MCP server. |
| **`face-recognition-tune`** | Companion patch implementing the `cpm tune` Controller Pattern — fine-tunes a **LoRA adapter** over the frozen base weights for domain adaptation. |

The weights are hosted at **`https://huggingface.co/CognitiveOS/vision`** and
declared in `cognitive.json` as `brain.wide_model.weights.remote`
(`source: huggingface`, `model_id: CognitiveOS/vision`), so `cpm install`
downloads the GGUF at install time and the daemon registers the model for
`wide_model_load`. The recognition backbone is pretrained (VGGFace2); learning
**who is who** happens through enrollment (embedding gallery), not retraining.

## cpm implementation notes (findings)

Findings from building this package against the current `cpm` implementation
(`cpm@v1.0.9-alpha`, built from `CognitiveOS-Project/cpm`) and the draft specs
(`manifest-fields.md`, `cgp-format.md`, `ai-model-publisher.md`). The spec docs
and the implementation are not always in sync — the implementation wins for
what `cpm pack` accepts.

- **`brain.wide_model.routing`** — the spec (`manifest-fields.md`) defines an
  object `{ model_id, tags }`; the cpm Go struct defines an **array** of
  `{ capability, priority }`. Divergent semantics, so `routing` is **omitted**
  from this manifest. The daemon's model-registry wiring described in
  `raw-model.md` (§Multi-Model Routing) is deferred until the shapes align.
- **`dependencies`** — cpm's schema validator **rejects the field for any
  value** (`ERROR:V006 ... invalid jsonType map[string]string`), including an
  empty `{}`. It is therefore omitted; package relationships are documented in
  the READMEs. Re-add when cpm's validator accepts the field.
- **`brain.training`** — used by `face-recognition-tune`; documented in the
  `cpm-tune.md` tutorial but **not present in either schema** (local mirror or
  cpm's bundled one). It passes because unknown properties are allowed, but it
  is not schema-validated.
- **Remote weights at pack time** — `cpm pack` resolves `weights.remote`
  entries to an **optional local file `weights/<filename>`**; a missing file
  prints `Warning: referenced file ... not found, skipping` and the archive is
  built without weights. That is correct for remote-only weights (downloaded
  by the daemon at install from `CognitiveOS/vision`).
- **Tools exec bit** — plain `cpm pack` stores `tools/` files at `0644`.
  Use **`cpm pack --bin tools`** so archive tools keep `0755` (the MCP servers
  in this repo are shebang scripts that must be executable).
- **HF model card** — use a **generic card** (`library_name: gguf`, tags like
  `face-recognition`, `face-detection`, `vision`) for `CognitiveOS/vision`,
  **not** the Diffusion/LoRA template (that metadata describes diffusers LoRA
  adapters with `lora_*` keys; these are full CNNs). The GGUF uses a generic
  `general.architecture`, so llama.cpp cannot load it yet — that is the
  architecture-reconstruction follow-up.


## Input models

| Model | Manifest | Tensors | Use |
|-------|----------|---------|-----|
| `tiny_face_detector_model` | `tiny_face_detector_model-weights_manifest.json` | 19 | Tiny face detection |
| `face_landmark_68_model` | `face_landmark_68_model-weights_manifest.json` | 49 | 68-point landmarks |
| `face_landmark_68_tiny_model` | `face_landmark_68_tiny_model-weights_manifest.json` | 28 | Tiny 68-point landmarks |
| `face_recognition_model` | `face_recognition_model-weights_manifest.json` | 117 | Face embedding |
| `ssd_mobilenetv1_model` | `ssd_mobilenetv1_model-weights_manifest.json` | 151 | SSD face detection |
| `age_gender_model` | `age_gender_model-weights_manifest.json` | 51 | Age / gender |
| `face_expression_model` | `face_expression_model-weights_manifest.json` | 49 | Facial expression |

Source: `https://github.com/vladmandic/face-api` at commit
`189226d63aabb48cb40776fd1c453ebc0fa722f1` (see `NOTICE`).

## Artifacts (per model, under `artifacts/<model>/`)

| File | Contents |
|------|----------|
| `<model>.safetensors` | All tensors, name → tensor, float32 (few int32 constants) |
| `<model>.safetensors.mapping.json` | name, shape, dtype, data_offsets, size, original TF.js quantization block |
| `<model>.gguf` | Same tensors packed in GGUF v3 format (alignment 32) |
| `<model>.gguf-mapping.json` | name, shape, ggml_type, data_offset, size, original quantization block |
| `mapping.json` | Intermediate TF.js extraction mapping (npy filenames + quantization) |
| `weights/*.npy` | Per-tensor float32 NPY files (intermediate) |

Tensor names keep the original TF.js names (e.g. `entry_flow/conv_in/filters`),
so every artifact is traceable to the manifest entry that produced it. All
tensors are **float32** (the dequantized values as produced by the TF.js
loader); the 15 scalar `int32` constant tensors in `ssd_mobilenetv1_model`
(postprocessor slice/shape params) are preserved as `int32`.

## Pipeline

```
upstream/face-api/model/*.bin + *-weights_manifest.json
        │  scripts/extract-tfjs-weights.js  (tf.io.weightsLoaderFactory, pure-JS tfjs)
        ▼
artifacts/<model>/weights/*.npy  +  mapping.json
        │  scripts/verify_tfjs_bin.py     (independent .bin parser, bitwise check)
        ▼
scripts/convert-to-safetensors.py ──► <model>.safetensors + .mapping.json  (roundtrip bitwise)
scripts/convert-to-gguf.py        ──► <model>.gguf + .gguf-mapping.json
scripts/read-gguf.py --all        ──► bitwise check gguf == safetensors
scripts/check_gguf_external.py    ──► spec check via external ggml/llama.cpp `gguf` reader
```

## Reproduce locally

Prerequisites: `node >= 18` with `npm`, and `python3 >= 3.10` with `pip`.

```bash
# 1. Clone the upstream repo at the pinned commit
mkdir -p upstream && git init upstream/face-api
git -C upstream/face-api remote add origin https://github.com/vladmandic/face-api
git -C upstream/face-api fetch --depth 1 origin 189226d63aabb48cb40776fd1c453ebc0fa722f1
git -C upstream/face-api checkout FETCH_HEAD

# 2. Python env
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt

# 3. Node env
( cd scripts && npm install @tensorflow/tfjs )

# 4. Run the full pipeline + validation
PYTHON=.venv/bin/python sh scripts/run_all_validations.sh
```

## Reproduce with Docker

```bash
docker build -t face-recognition-convert .
id=$(docker create face-recognition-convert)
docker cp $id:/artifacts ./artifacts-out
docker rm $id
```

The build clones face-api at the pinned commit, extracts, converts, and runs
the whole validation suite (log written to `artifacts/VALIDATION.log`).

## Validation

`scripts/run_all_validations.sh` runs, for **every model**, and fails (nonzero
exit) on any deviation:

1. TF.js extraction is bitwise-verified against an **independent** `.bin`
   parser (`verify_tfjs_bin.py`) that reimplements the tfjs dequantization
   (`v * scale + min` for uint8/uint16 → float32; `Math.round(...)` for int32).
2. Safetensors roundtrip: re-opening the `.safetensors` yields arrays that are
   bitwise identical to the extracted npy.
3. GGUF roundtrip via the reference loader `read-gguf.py`.
4. GGUF spec compliance via the **external** `gguf` reader (ggml/llama.cpp
   ecosystem).

Current status: **464/464 tensors across 7 models bitwise-verified in all
stages.**

## gguf consumer

No specific runtime was given, so the gguf files use the **generic packing**
(metadata keys `general.architecture`, `general.name`, `general.source`,
`general.source.commit`, `general.alignment`, `general.file_type`; tensor dims
stored in GGML/reversed order; `GGML_TYPE_F32`=0 and `GGML_TYPE_I32`=26).
`scripts/read-gguf.py` is the reference loader. If a specific ggml/gguf
consumer is chosen later, only the metadata keys (not the tensor payload) need
to be adapted.

## Architecture reconstruction (follow-up)

The conversion is weights-only. Reimplementing the networks from the TS sources
(`src/xception`, `src/ageGenderNet`, `src/tinyYolov2`, `src/faceLandmarkNet`,
`src/faceRecognitionNet`, `src/ssdMobilenetv1`, `src/tinyFaceDetector`) in
PyTorch/Keras and loading these tensors (with the documented transposes) so the
converted models can be run and compared against the original JS inference is
the next phase. The mapping JSONs carry everything needed for that wiring.

## License

The conversion code in this package is MIT. The converted weights are derived
from the face-api models, which are distributed under their own terms — see
`NOTICE` for the upstream source, commit OID, and license attribution.

## Author

Built and maintained by [jeanmachuca](https://github.com/jeanmachuca).

Support the author: <https://github.com/sponsors/jeanmachuca>
