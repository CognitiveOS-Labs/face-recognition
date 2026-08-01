#!/usr/bin/env python3
"""
convert-to-safetensors.py — pack the extracted TF.js tensors into safetensors.

Input per model (from scripts/extract-tfjs-weights.js):
  artifacts/<model>/mapping.json
  artifacts/<model>/weights/*.npy

Output:
  artifacts/<model>/<model>.safetensors
  artifacts/<model>/<model>.safetensors.mapping.json
    (name -> shape, dtype, data_offsets, size_bytes, original quantization)

Tensor keys keep the original TF.js weight names verbatim (traceability).
A roundtrip check re-opens the .safetensors and asserts bitwise equality with
the source .npy arrays.

Usage:
  python convert-to-safetensors.py [--artifacts <dir>] [--model <name>|all]
"""
import argparse
import json
import os
import sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

TENSOR_DTYPES = {
    np.dtype(np.float32): 'F32',
    np.dtype(np.float64): 'F64',
    np.dtype(np.float16): 'F16',
    np.dtype(np.int8): 'I8',
    np.dtype(np.int16): 'I16',
    np.dtype(np.int32): 'I32',
    np.dtype(np.int64): 'I64',
    np.dtype(np.uint8): 'U8',
    np.dtype(np.bool_): 'BOOL',
}


def load_tensors(mdir, mapping):
    arrays = {}
    for t in mapping['tensors']:
        arr = np.load(os.path.join(mdir, t['file']))
        if arr.ndim == 0:
            arr = arr.reshape(1)
        arrays[t['name']] = arr
    return arrays


def roundtrip_check(path, expected):
    with safe_open(path, framework='np') as f:
        for name, arr in expected.items():
            got = f.get_tensor(name)
            if not np.array_equal(got, arr):
                return False, f'{name}: bitwise mismatch'
            assert tuple(got.shape) == tuple(arr.shape), (name, got.shape, arr.shape)
    return True, None


def convert_model(artifacts_dir, model):
    mdir = os.path.join(artifacts_dir, model)
    mapping = json.load(open(os.path.join(mdir, 'mapping.json')))
    arrays = load_tensors(mdir, mapping)

    st_path = os.path.join(mdir, f'{model}.safetensors')
    save_file(arrays, st_path)

    # Read back the offsets written by safetensors for the mapping.
    offsets = {}
    with open(st_path, 'rb') as f:
        n = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
        header = json.loads(f.read(n).decode('utf-8'))
        data_start = 8 + n
    for name, meta in header.items():
        b, e = meta['data_offsets']
        offsets[name] = [data_start + b, data_start + e]

    ok, err = roundtrip_check(st_path, arrays)
    if not ok:
        raise RuntimeError(f'{model}: roundtrip failed: {err}')

    st_tensors = []
    for t in mapping['tensors']:
        arr = arrays[t['name']]
        st_tensors.append({
            'name': t['name'],
            'shape': list(arr.shape),
            'original_shape': list(t['shape']),
            'dtype': TENSOR_DTYPES.get(arr.dtype, str(arr.dtype)),
            'data_offsets': offsets[t['name']],
            'size_bytes': int(arr.nbytes),
            'quantization': t.get('quantization'),
            'source_dtype': t.get('source_dtype'),
        })

    st_mapping = {
        'model': model,
        'format': 'safetensors',
        'source': mapping.get('source'),
        'tensor_count': len(st_tensors),
        'total_bytes': int(sum(t['size_bytes'] for t in st_tensors)),
        'tensors': st_tensors,
    }
    st_map_path = os.path.join(mdir, f'{model}.safetensors.mapping.json')
    with open(st_map_path, 'w') as f:
        json.dump(st_mapping, f, indent=2)

    print(f'✓ {model}: {len(st_tensors)} tensors -> '
          f'{os.path.relpath(st_path)} (roundtrip bitwise OK)')
    return st_path, st_map_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts', default=os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
    ap.add_argument('--model', default='all')
    args = ap.parse_args()
    artifacts_dir = os.path.abspath(args.artifacts)

    models = [args.model] if args.model != 'all' else sorted(
        d for d in os.listdir(artifacts_dir)
        if os.path.isdir(os.path.join(artifacts_dir, d)))
    results = {}
    for m in models:
        try:
            results[m] = convert_model(artifacts_dir, m)
        except Exception as e:
            print(f'✗ {m}: {e}')
            results[m] = None
    failed = [m for m, r in results.items() if r is None]
    if failed:
        sys.exit(1)
    print(f'\nConverted {len(models)} models to safetensors.')


if __name__ == '__main__':
    main()
