---
license: mit
tags:
  - face-recognition
  - face-detection
  - face-landmarks
  - face-expression
  - face-age-gender
  - gguf
  - vision
library_name: gguf
pipeline_tag: image-classification
---

# face-api weights (gguf + safetensors)

Pre-trained model weights from [vladmandic/face-api](https://github.com/vladmandic/face-api)
at commit `189226d63aabb48cb40776fd1c453ebc0fa722f1`, converted losslessly to
portable **safetensors** and **gguf** (float32) formats by the
[`CognitiveOS-Labs/face-recognition`](https://github.com/CognitiveOS-Labs/face-recognition)
reproducible pipeline.

## Models

| File | Tensors | Purpose |
|------|---------|---------|
| `tiny_face_detector_model.gguf` | 19 | Tiny face detection |
| `face_landmark_68_model.gguf` | 49 | 68-point landmarks |
| `face_landmark_68_tiny_model.gguf` | 28 | Tiny 68-point landmarks |
| `face_recognition_model.gguf` | 117 | Face embedding (128-d) |
| `ssd_mobilenetv1_model.gguf` | 151 | SSD face detection |
| `age_gender_model.gguf` | 51 | Age / gender |
| `face_expression_model.gguf` | 49 | Facial expression |

Every model is also provided as `.safetensors` and both formats carry a
`*-mapping.json` recording each tensor's name, shape, dtype, data location,
and the original TF.js quantization block (source traceability).

## Format notes

- **Weights only.** The face-api repo has no `model.json` — the network
  topologies live in its TypeScript sources under `src/`. These artifacts
  preserve the original TF.js tensor names verbatim.
- All tensors are **float32** (the dequantized values produced by the TF.js
  loader), except 15 scalar `int32` constants in `ssd_mobilenetv1_model`
  (postprocessor shape parameters).
- The gguf files use **generic packing** (metadata keys
  `general.architecture`, `general.name`, `general.source`,
  `general.source.commit`, `general.alignment`, `general.file_type`; GGML
  dim order; `GGML_TYPE_F32`=0, `GGML_TYPE_I32`=26). They are structurally
  valid GGUF v3 but use a custom `general.architecture`, so llama.cpp will not
  load them yet — a runtime for these architectures is the documented
  follow-up (see the `face-recognition` CGP repo).
- Reference loaders and the full bitwise-validation suite live in the
  `face-recognition` CGP repo under `scripts/`.

## Consumed by

- [`CognitiveOS-Labs/face-recognition`](https://github.com/CognitiveOS-Labs/face-recognition)
  — CognitiveOS patch declaring `face_recognition_model.gguf` as its remote
  wide-model weight (downloaded at `cpm install` time).
- [`CognitiveOS-Labs/face-recognition-tune`](https://github.com/CognitiveOS-Labs/face-recognition-tune)
  — `cpm tune` LoRA trainer over the frozen base weights.

## Source & license

Weights derived from `vladmandic/face-api` (MIT), commit
`189226d63aabb48cb40776fd1c453ebc0fa722f1`, provided "as-is". Conversion tooling
in the `face-recognition` CGP repo is MIT.

The recognition backbone was trained on VGGFace2-like data by the face-api
project; generic face recognition works out of the box, while recognizing
specific people is an enrollment (embedding-gallery) operation.
