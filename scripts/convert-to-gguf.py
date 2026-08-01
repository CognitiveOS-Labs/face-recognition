#!/usr/bin/env python3
"""
convert-to-gguf.py — pack the extracted TF.js tensors into a generic GGUF file.

GGUF spec (v3): https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

Layout:
  magic "GGUF" | uint32 version | uint64 tensor_count | uint64 metadata_kv_count
  metadata KV pairs (key string, value type uint32, value)
  tensor infos: name string, uint32 n_dims, n_dims × uint64 dims (GGML/reverse
    order), uint32 ggml_type, uint64 offset
  data section: per-tensor blobs, each aligned to general.alignment (default 32)

Tensor dims are stored reversed (GGML order) per the spec; numpy/python order
is recovered by reversing again. All tensors are float32 (GGML_TYPE_F32=0),
except the int32 constant tensors (GGML_TYPE_I32=26).

Output:
  artifacts/<model>/<model>.gguf
  artifacts/<model>/<model>.gguf-mapping.json

No target runtime was specified, so this is the generic packing; a reference
loader is provided in scripts/read-gguf.py and roundtrip equality with the
safetensors arrays is asserted.

Usage:
  python convert-to-gguf.py [--artifacts <dir>] [--model <name>|all]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_I32 = 26

# GGUFValueType
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

ALIGNMENT = 32

NUMPY_TO_GGML = {
    np.dtype(np.float32): GGML_TYPE_F32,
    np.dtype(np.int32): GGML_TYPE_I32,
}


class Writer:
    def __init__(self):
        self.buf = bytearray()

    def write(self, b):
        self.buf += b

    def u8(self, v):
        self.write(struct.pack('<B', v))

    def u32(self, v):
        self.write(struct.pack('<I', v))

    def u64(self, v):
        self.write(struct.pack('<Q', v))

    def string(self, s):
        b = s.encode('utf-8')
        self.u64(len(b))
        self.write(b)

    def kv_string(self, key, value):
        self.string(key)
        self.u32(GGUF_TYPE_STRING)
        self.string(value)

    def kv_u32(self, key, value):
        self.string(key)
        self.u32(GGUF_TYPE_UINT32)
        self.u32(value)


def tensor_byte_size(arr):
    n = int(np.prod(arr.shape)) if arr.size else 0
    return n * arr.itemsize


def convert_model(artifacts_dir, model):
    mdir = os.path.join(artifacts_dir, model)
    mapping = json.load(open(os.path.join(mdir, 'mapping.json')))
    source = mapping.get('source', {})

    tensors = []
    arrays = {}
    for t in mapping['tensors']:
        arr = np.load(os.path.join(mdir, t['file']))
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.dtype not in NUMPY_TO_GGML:
            raise RuntimeError(f'{model}/{t["name"]}: unsupported dtype {arr.dtype}')
        arrays[t['name']] = arr
        tensors.append(t)

    w = Writer()
    w.write(GGUF_MAGIC)
    w.u32(GGUF_VERSION)
    w.u64(len(tensors))
    w.u64(5)  # metadata_kv_count

    w.kv_string('general.architecture', 'face-api')
    w.kv_string('general.name', model)
    w.kv_string('general.source', source.get('repo', ''))
    w.kv_string('general.source.commit', source.get('commit', ''))
    w.kv_u32('general.alignment', ALIGNMENT)

    # Compute data offsets (aligned) before writing the data section.
    offset = 0
    offsets = {}
    for t in tensors:
        arr = arrays[t['name']]
        offsets[t['name']] = offset
        offset += tensor_byte_size(arr)
        offset += (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT

    # Tensor infos (dims in GGML/reverse order).
    for t in tensors:
        arr = arrays[t['name']]
        shape = list(arr.shape)
        w.string(t['name'])
        w.u32(len(shape))
        for d in reversed(shape):
            w.u64(d)
        w.u32(NUMPY_TO_GGML[arr.dtype])
        w.u64(offsets[t['name']])

    # Data section: must start on an ALIGNMENT boundary (absolute file offset).
    # Each tensor is then placed at offsets aligned to ALIGNMENT, relative to
    # the data section start.
    while len(w.buf) % ALIGNMENT != 0:
        w.write(b'\x00')
    data_start = len(w.buf)
    for t in tensors:
        arr = arrays[t['name']]
        while (len(w.buf) - data_start) % ALIGNMENT != 0:
            w.write(b'\x00')
        if len(w.buf) - data_start != offsets[t['name']]:
            raise RuntimeError(f'{t["name"]}: offset mismatch '
                               f'{offsets[t["name"]]} != {len(w.buf) - data_start}')
        w.write(bytes(arr.tobytes()))
    while len(w.buf) % ALIGNMENT != 0:
        w.write(b'\x00')

    with open(os.path.join(mdir, f'{model}.gguf'), 'wb') as f:
        f.write(w.buf)

    ggml_type_names = {GGML_TYPE_F32: 'F32', GGML_TYPE_I32: 'I32'}
    gg_tensors = []
    for t in tensors:
        arr = arrays[t['name']]
        gg_tensors.append({
            'name': t['name'],
            'shape': list(arr.shape),
            'original_shape': list(t['shape']),
            'dtype': ggml_type_names[NUMPY_TO_GGML[arr.dtype]],
            'ggml_type': NUMPY_TO_GGML[arr.dtype],
            'data_offset': offsets[t['name']],
            'size_bytes': int(arr.nbytes),
            'quantization': t.get('quantization'),
            'source_dtype': t.get('source_dtype'),
        })

    gg_mapping = {
        'model': model,
        'format': 'gguf',
        'gguf_version': GGUF_VERSION,
        'alignment': ALIGNMENT,
        'source': source,
        'tensor_count': len(gg_tensors),
        'total_bytes': int(sum(t['size_bytes'] for t in gg_tensors)),
        'data_start': data_start,
        'tensors': gg_tensors,
    }
    with open(os.path.join(mdir, f'{model}.gguf-mapping.json'), 'w') as f:
        json.dump(gg_mapping, f, indent=2)

    print(f'✓ {model}: {len(gg_tensors)} tensors -> '
          f'{os.path.relpath(os.path.join(mdir, f"{model}.gguf"))}')
    return os.path.join(mdir, f'{model}.gguf')


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
    print(f'\nConverted {len(models)} models to gguf.')


if __name__ == '__main__':
    main()
