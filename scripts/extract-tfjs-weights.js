#!/usr/bin/env node
/*
 * extract-tfjs-weights.js
 *
 * Extracts named, dequantized tensors from TF.js weight manifests + .bin files
 * using tf.io.weightsLoaderFactory (canonical tfjs path). Outputs, per model:
 *
 *   <out>/<model-name>/mapping.json
 *   <out>/<model-name>/weights/<sanitized-tensor-name>.npy
 *
 * Tensor values are written in float32 (or int32, for the few int32 constant
 * tensors, e.g. ssd_mobilenetv1 postprocessor params) as NPY v1.0 files.
 *
 * Usage:
 *   node extract-tfjs-weights.js [--model-dir <dir>] [--out <dir>]
 *                                [--manifest <file>] [--commit <oid>] [--repo <url>]
 *
 * Defaults:
 *   --model-dir  ../upstream/face-api/model
 *   --out        ../artifacts
 *
 * Source: vladmandic/face-api (https://github.com/vladmandic/face-api)
 */
'use strict';

const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs');

const COMMIT = process.env.FACE_API_COMMIT || '189226d63aabb48cb40776fd1c453ebc0fa722f1';
const REPO = process.env.FACE_API_REPO || 'https://github.com/vladmandic/face-api';

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next != null && !next.startsWith('--')) {
        args[key] = next;
        i++;
      } else {
        args[key] = true;
      }
    }
  }
  return args;
}

function sanitize(name) {
  return name.replace(/[^A-Za-z0-9_.\-]/g, '_');
}

function padHeader(header) {
  // npy v1 header = 6 magic + 2 version + 2 length = 10 fixed bytes; the dict
  // must be padded so (10 + dictLen) is a multiple of 16.
  const prefixLen = 10;
  const padded = 16 - ((prefixLen + header.length) % 16);
  return header + ' '.repeat(padded === 16 ? 0 : padded);
}

function writeNpy(filePath, data, shape, dtype) {
  let descr, arr;
  if (dtype === 'float32') {
    descr = '<f4';
    arr = new Float32Array(data);
  } else if (dtype === 'int32') {
    descr = '<i4';
    arr = new Int32Array(data);
  } else if (dtype === 'int16') {
    descr = '<i2';
    arr = new Int16Array(data);
  } else if (dtype === 'uint8') {
    descr = '|u1';
    arr = new Uint8Array(data);
  } else {
    throw new Error(`Unsupported npy dtype: ${dtype}`);
  }

  const shapeStr = '(' + shape.join(', ') + (shape.length === 1 ? ',' : '') + ')';
  const header = padHeader(
    `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shapeStr}, }`
  );
  const fixedPrefix = 10; // magic(6) + version(2) + header length(2)
  const buf = Buffer.alloc(fixedPrefix + header.length + arr.byteLength);
  buf.write('\x93NUMPY', 0, 6, 'latin1');
  buf[6] = 1; // major version
  buf[7] = 0; // minor version
  buf.writeUInt16LE(header.length, 8);
  buf.write(header, 10, 'latin1');
  Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).copy(buf, fixedPrefix + header.length);
  fs.writeFileSync(filePath, buf);
}

async function processManifest(manifestPath, modelName, modelDir, outDir) {
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

  // Canonical tfjs loader: reads .bin files from disk, dequantizes.
  const fetchWeightsFromDisk = (filePaths) =>
    Promise.all(filePaths.map((fp) => {
      const resolved = path.isAbsolute(fp) ? fp : path.join(modelDir, fp);
      return fs.promises.readFile(resolved).then((b) => b.buffer);
    }));
  const loadWeights = tf.io.weightsLoaderFactory(fetchWeightsFromDisk);
  const weightMap = await loadWeights(manifest, modelDir + path.sep);

  const modelOut = path.join(outDir, modelName);
  const weightsDir = path.join(modelOut, 'weights');
  fs.mkdirSync(weightsDir, { recursive: true });

  const tensors = [];
  const names = Object.keys(weightMap);
  for (const name of names) {
    const tensor = weightMap[name];
    const shape = tensor.shape;
    const dtype = tensor.dtype; // 'float32' | 'int32' | ...
    const values = tensor.dataSync();
    const fileName = sanitize(name) + '.npy';
    const filePath = path.join(weightsDir, fileName);
    writeNpy(filePath, values, shape, dtype);

    // Locate the manifest entry to preserve the original quantization block.
    let quantization = null;
    let sourceDtype = null;
    outer:
    for (const group of manifest) {
      for (const entry of group.weights) {
        if (entry.name === name) {
          quantization = entry.quantization || null;
          sourceDtype = entry.dtype || null;
          break outer;
        }
      }
    }

    tensors.push({
      name,
      shape,
      dtype,
      file: 'weights/' + fileName,
      numel: shape.reduce((a, b) => a * b, 1),
      size_bytes: values.byteLength,
      quantization,
      source_dtype: sourceDtype,
    });
    tensor.dispose();
  }

  tensors.sort((a, b) => (a.name < b.name ? -1 : 1));

  const mapping = {
    model: modelName,
    format: 'tfjs-extracted-float32',
    source: { repo: REPO, commit: COMMIT },
    manifest: path.basename(manifestPath),
    tensor_count: tensors.length,
    total_bytes: tensors.reduce((s, t) => s + t.size_bytes, 0),
    tensors,
  };

  fs.writeFileSync(
    path.join(modelOut, 'mapping.json'),
    JSON.stringify(mapping, null, 2)
  );

  console.log(`✓ ${modelName}: ${tensors.length} tensors, ` +
    `${(mapping.total_bytes / 1024).toFixed(1)} KiB -> ${path.relative(process.cwd(), modelOut)}`);
  return mapping;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const modelDir = path.resolve(args['model-dir'] || path.join(__dirname, '..', 'upstream', 'face-api', 'model'));
  const outDir = path.resolve(args['out'] || path.join(__dirname, '..', 'artifacts'));
  fs.mkdirSync(outDir, { recursive: true });

  let manifests;
  if (args['manifest']) {
    manifests = [path.resolve(args['manifest'])];
  } else {
    manifests = fs.readdirSync(modelDir)
      .filter((f) => f.endsWith('-weights_manifest.json'))
      .sort()
      .map((f) => path.join(modelDir, f));
  }
  if (manifests.length === 0) {
    console.error(`No *-weights_manifest.json found in ${modelDir}`);
    process.exit(1);
  }

  const models = [];
  for (const mf of manifests) {
    const base = path.basename(mf);
    const modelName = base.replace(/-weights_manifest\.json$/, '');
    models.push(await processManifest(mf, modelName, modelDir, outDir));
  }

  console.log(`\nDone: extracted ${models.length} models into ${outDir}`);
}

main().catch((err) => {
  console.error('Extraction failed:', err);
  process.exit(1);
});
