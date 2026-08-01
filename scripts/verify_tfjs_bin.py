#!/usr/bin/env python3
"""
verify_tfjs_bin.py — independent verification of the Node extraction.

Parses the raw TF.js .bin files directly (flat concatenated tensor data,
dequantization matching tfjs-core `decodeWeight`) and compares each tensor
bitwise against the .npy files produced by scripts/extract-tfjs-weights.js.

The tfjs on-disk format is: for each weightSpec in manifest order, within each
manifest group the tensor data is stored back-to-back. Byte length per tensor =
numel * bytesPerDtype, where bytesPerDtype uses the quantization dtype (uint8=1,
uint16=2, float16=2) when a quantization block is present, else the tensor
dtype (float32=4, int32=4). Dequantization mirrors tfjs-core:

  float32 <- uint8/uint16 : value = v * scale + min            (float64 math, stored float32)
  int32   <- uint8/uint16 : value = Math.round(v * scale + min)   (floor(v*scale+min+0.5))
  float32 <- float16      : decode float16 -> float32

Usage:
  python verify_tfjs_bin.py [--artifacts <dir>] [--model-dir <dir>]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

DTYPE_BYTES = {
    'uint8': 1, 'uint16': 2, 'float16': 2,
    'float32': 4, 'int32': 4, 'int16': 2, 'int8': 1,
    'bool': 1, 'complex64': 8, 'string': 2,
}

NUMPY_DTYPE = {
    'float32': np.float32, 'int32': np.int32,
    'int16': np.int16, 'uint8': np.uint8,
}


def load_bin(model_dir, group):
    parts = []
    for p in group.get('paths', []):
        with open(os.path.join(model_dir, p), 'rb') as f:
            parts.append(f.read())
    return b''.join(parts)


def dequantize(raw, spec):
    """Return a 1-D typed array matching tfjs-core decodeWeight semantics."""
    size = int(np.prod(spec['shape'])) if spec['shape'] else 1
    q = spec.get('quantization')
    target = spec['dtype']
    if q is not None:
        qdtype = q['dtype']
        if qdtype in ('uint8', 'uint16'):
            dtype = np.uint8 if qdtype == 'uint8' else np.uint16
            arr = np.frombuffer(raw, dtype=dtype, count=size)
            if target == 'float32':
                return (arr.astype(np.float64) * q['scale'] + q['min']).astype(np.float32)
            if target == 'int32':
                # JS Math.round: floor(x + 0.5)
                return np.floor(arr.astype(np.float64) * q['scale'] + q['min'] + 0.5).astype(np.int32)
            raise ValueError(f'unhandled quant target {target}')
        if qdtype == 'float16':
            if target != 'float32':
                raise ValueError('float16 quantization only supports float32')
            return np.frombuffer(raw, dtype=np.float16, count=size).astype(np.float32)
        raise ValueError(f'unhandled quantization dtype {qdtype}')
    dtype = NUMPY_DTYPE.get(target)
    if dtype is None:
        raise ValueError(f'unhandled dtype {target}')
    return np.frombuffer(raw, dtype=dtype, count=size)


def verify_model(artifacts_dir, model_dir, model_name):
    mdir = os.path.join(artifacts_dir, model_name)
    mapping = json.load(open(os.path.join(mdir, 'mapping.json')))
    manifest_path = os.path.join(model_dir, mapping['manifest'])
    manifest = json.load(open(manifest_path))
    manifest = manifest if isinstance(manifest, list) else [manifest]

    expected = {}
    for t in mapping['tensors']:
        expected[t['name']] = t

    checked = 0
    mismatches = []
    for group in manifest:
        data = load_bin(model_dir, group)
        offset = 0
        for spec in group['weights']:
            name = spec['name']
            size = int(np.prod(spec['shape'])) if spec['shape'] else 1
            q = spec.get('quantization')
            bytes_per = DTYPE_BYTES.get(q['dtype'] if q else spec['dtype'])
            if bytes_per is None:
                raise ValueError(f'no byte size for {name}')
            byte_len = size * bytes_per
            raw = data[offset:offset + byte_len]
            offset += byte_len
            vals = dequantize(raw, spec)
            arr = vals.reshape(spec['shape'])

            meta = expected.get(name)
            if meta is None:
                mismatches.append((name, 'missing from mapping'))
                continue
            npy_path = os.path.join(mdir, meta['file'])
            npy = np.load(npy_path)
            if not np.array_equal(arr, npy):
                mismatches.append((name, f'bitwise mismatch '
                    f'{spec["shape"]} vs {meta["shape"]}'))
            checked += 1

    if checked != len(expected):
        mismatches.append((f'mapping has {len(expected)} tensors but '
                           f'manifest declared {checked}'))

    if mismatches:
        print(f'✗ {model_name}: FAIL')
        for m in mismatches:
            print(f'    {m[0]}: {m[1]}')
        return False
    print(f'✓ {model_name}: {checked} tensors bitwise-verified '
          f'({mapping["tensor_count"]} in mapping)')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts', default=os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
    ap.add_argument('--model-dir', default=os.path.join(os.path.dirname(__file__), '..', 'upstream', 'face-api', 'model'))
    ap.add_argument('--model', default=None, help='verify a single model')
    args = ap.parse_args()

    artifacts_dir = os.path.abspath(args.artifacts)
    model_dir = os.path.abspath(args.model_dir)
    models = [args.model] if args.model else sorted(os.listdir(artifacts_dir))
    models = [m for m in models if os.path.isdir(os.path.join(artifacts_dir, m))]

    results = {m: verify_model(artifacts_dir, model_dir, m) for m in models}
    failed = [m for m, ok in results.items() if not ok]
    if failed:
        print(f'\nFAILED: {failed}')
        sys.exit(1)
    print(f'\nAll {len(models)} models verified bitwise.')


if __name__ == '__main__':
    main()
