# face-recognition

A **Cognitive Patch** (`.cgp`) for **CognitiveOS** — an independent operating
system where the AI is the OS. See [cognitive-os.org](https://cognitive-os.org).

Provides face **detection, landmark alignment, recognition (128-d
embeddings), age/gender, expression**, and **enrollment of known identities**
(learning "who is who" via an embedding gallery).

The weights are **pre-trained** (converted from
[vladmandic/face-api](https://github.com/vladmandic/face-api) @ commit
`189226d63aabb48cb40776fd1c453ebc0fa722f1`, VGGFace2-trained recognition
backbone) and **not bundled in this archive** — they are declared as a remote
Hugging Face model and downloaded at `cpm install` time.

## Package architecture

This patch is one of two that share the converted weights:

| Package | Role |
|---------|------|
| **`face-recognition`** (this repo) | Installs the pretrained weights (declared as a remote Hugging Face model) and exposes face **detection / recognition / enrollment** tools via an MCP server. |
| **`face-recognition-tune`** | Companion patch implementing the `cpm tune` Controller Pattern — fine-tunes a **LoRA adapter** over the frozen base weights for domain adaptation. |

The weights are hosted at **`https://huggingface.co/CognitiveOS/vision`** and
declared in `cognitive.json` as `brain.wide_model.weights.remote`
(`source: huggingface`, `model_id: CognitiveOS/vision`,
`filename: face_recognition_model.gguf`, with `size_bytes` and `sha256` set),
so `cpm install` downloads the GGUF, verifies it, and the daemon registers the
model for `wide_model_load`. The recognition backbone is pretrained; learning
**who is who** happens through enrollment (embedding gallery), not retraining.

The full 7-model set (detector, landmarks, embedding, age/gender, expression)
is hosted on the same HF repo; the MCP server consumes them in the
architecture-reconstruction phase.

## Model weights

| File (on `CognitiveOS/vision`) | Tensors | Purpose |
|-------|----------|---------|
| `tiny_face_detector_model.gguf` | 19 | Tiny face detection |
| `face_landmark_68_model.gguf` | 49 | 68-point landmarks |
| `face_landmark_68_tiny_model.gguf` | 28 | Tiny 68-point landmarks |
| `face_recognition_model.gguf` | 117 | Face embedding (128-d) |
| `ssd_mobilenetv1_model.gguf` | 151 | SSD face detection |
| `age_gender_model.gguf` | 51 | Age / gender |
| `face_expression_model.gguf` | 49 | Facial expression |

Each is also provided as `.safetensors`, both with a `*-mapping.json`
recording tensor name, shape, dtype, data location, and the original TF.js
quantization block.

**How they were produced:** the conversion pipeline is a standalone,
reproducible tool — [`CognitiveOS-Labs/tfjs-weights-to-gguf`](https://github.com/CognitiveOS-Labs/tfjs-weights-to-gguf).
It converts the TF.js weight blobs (`*-weights_manifest.json` + `.bin`) into
npy → safetensors → gguf, bitwise-verified at every stage. `CognitiveOS/vision`
is its published output.

## Tools (MCP)

`tools/mcp-face-server` (stdio, JSON-lines per `mcp-conventions.md`) exposes:

- `cognitiveos.face.detect` — faces + bounding boxes + keypoints
- `cognitiveos.face.embed` — 128-d embedding for an aligned face
- `cognitiveos.face.enroll` — store a labeled embedding in the gallery
- `cognitiveos.face.match` — cosine-distance match against the gallery

> **Status:** structural stub. Tool metadata, protocol shape, and error
> envelope (`ERROR:<code>:<message>`) are in place; tool calls return
> `E_INTERNAL` until the CNN runtimes land in the architecture-reconstruction
> phase.

## cpm implementation notes (findings)

Findings from building against the current `cpm` implementation
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
  Use **`cpm pack --bin tools`** so archive tools keep `0755` (the MCP server
  in this repo is a shebang script that must be executable).

## Architecture reconstruction (follow-up)

The conversion is weights-only (no `model.json`; topologies live in the
face-api TS sources). Reimplementing the networks
(`src/xception`, `src/ageGenderNet`, `src/tinyYolov2`,
`src/faceLandmarkNet`, `src/faceRecognitionNet`, `src/ssdMobilenetv1`,
`src/tinyFaceDetector`) in PyTorch/Keras, loading these tensors, and wiring
them into the MCP server so inference actually runs is the next phase. The
mapping JSONs on `CognitiveOS/vision` carry everything needed for that wiring.

## Build

```bash
cpm pack --bin tools   # bundles manifest + prompts + tools/ (exec bit preserved)
```

## License

MIT. The weights are derived from face-api under their own terms — upstream
attribution lives in the
[`tfjs-weights-to-gguf`](https://github.com/CognitiveOS-Labs/tfjs-weights-to-gguf)
`NOTICE` and on the `CognitiveOS/vision` model card.

## Author

Built and maintained by [jeanmachuca](https://github.com/jeanmachuca).

Support the author: <https://github.com/sponsors/jeanmachuca>
