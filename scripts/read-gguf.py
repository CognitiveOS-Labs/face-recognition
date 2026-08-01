#!/usr/bin/env python3
"""
read-gguf.py — reference GGUF loader (generic, spec v3).

Parses the header (magic, version, tensor count, metadata KV), tensor infos
(name, dims, ggml_type, offset), and slices the data section to reconstruct
numpy arrays. Can verify bitwise equality against the .safetensors or .npy
arrays of the same model.

Usage:
  python read-gguf.py <model.gguf> [--check-safetensors <model.safetensors>]
  python read-gguf.py --all [--artifacts <dir>]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

GGML_TYPE_F32 = 0
GGML_TYPE_I32 = 26

GGML_TYPE_SIZES = {GGML_TYPE_F32: 4, GGML_TYPE_I32: 4}
GGML_TYPE_NP = {GGML_TYPE_F32: np.float32, GGML_TYPE_I32: np.int32}
GGML_TYPE_NAMES = {GGML_TYPE_F32: 'F32', GGML_TYPE_I32: 'I32'}

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


class Reader:
    def __init__(self, buf):
        self.buf = buf
        self.off = 0

    def read(self, n):
        b = self.buf[self.off:self.off + n]
        if len(b) != n:
            raise ValueError('truncated gguf')
        self.off += n
        return b

    def u8(self):
        return self.read(1)[0]

    def u32(self):
        return struct.unpack('<I', self.read(4))[0]

    def u64(self):
        return struct.unpack('<Q', self.read(8))[0]

    def f32(self):
        return struct.unpack('<f', self.read(4))[0]

    def f64(self):
        return struct.unpack('<d', self.read(8))[0]

    def string(self):
        n = self.u64()
        return self.read(n).decode('utf-8')

    def value(self, ttype):
        if ttype == GGUF_TYPE_UINT8:
            return self.u8()
        if ttype == GGUF_TYPE_UINT16:
            return struct.unpack('<H', self.read(2))[0]
        if ttype == GGUF_TYPE_UINT32:
            return self.u32()
        if ttype == GGUF_TYPE_UINT64:
            return self.u64()
        if ttype == GGUF_TYPE_INT8:
            return struct.unpack('<b', self.read(1))[0]
        if ttype == GGUF_TYPE_INT16:
            return struct.unpack('<h', self.read(2))[0]
        if ttype == GGUF_TYPE_INT32:
            return struct.unpack('<i', self.read(4))[0]
        if ttype == GGUF_TYPE_INT64:
            return struct.unpack('<q', self.read(8))[0]
        if ttype == GGUF_TYPE_FLOAT32:
            return self.f32()
        if ttype == GGUF_TYPE_FLOAT64:
            return self.f64()
        if ttype == GGUF_TYPE_BOOL:
            return bool(self.u8())
        if ttype == GGUF_TYPE_STRING:
            return self.string()
        if ttype == GGUF_TYPE_ARRAY:
            elem_type = self.u32()
            n = self.u64()
            return [self.value(elem_type) for _ in range(n)]
        raise ValueError(f'unknown gguf value type {ttype}')


def load_gguf(path):
    buf = open(path, 'rb').read()
    r = Reader(buf)
    if r.read(4) != b'GGUF':
        raise ValueError('not a gguf file')
    version = r.u32()
    tensor_count = r.u64()
    kv_count = r.u64()

    metadata = {}
    for _ in range(kv_count):
        key = r.string()
        ttype = r.u32()
        metadata[key] = r.value(ttype)

    alignment = int(metadata.get('general.alignment', 32))

    infos = []
    for _ in range(tensor_count):
        name = r.string()
        n_dims = r.u32()
        dims = [r.u64() for _ in range(n_dims)]
        ggml_type = r.u32()
        offset = r.u64()
        infos.append((name, dims, ggml_type, offset))

    data_start = r.off
    # The data section starts on the next alignment boundary (absolute file
    # offset); tensor offsets are relative to it.
    pad = (alignment - (data_start % alignment)) % alignment
    data_start += pad
    data = buf[data_start:]

    tensors = {}
    for name, dims, ggml_type, offset in infos:
        dtype = GGML_TYPE_NP.get(ggml_type)
        if dtype is None:
            raise ValueError(f'unsupported ggml_type {ggml_type} for {name}')
        np_dtype = np.dtype(dtype)
        n = int(np.prod(dims))
        start = offset
        end = start + n * np_dtype.itemsize
        raw = data[start:end]
        arr = np.frombuffer(raw, dtype=np_dtype, count=n)
        # GGUF dims are in GGML/reverse order; numpy/python order is reversed.
        shape = tuple(reversed(dims))
        arr = arr.reshape(shape)
        tensors[name] = arr

    return {
        'version': version,
        'metadata': metadata,
        'tensor_count': tensor_count,
        'alignment': alignment,
        'data_start': data_start,
        'tensors': tensors,
    }


def check_against_safetensors(gguf_path, st_path):
    gguf = load_gguf(gguf_path)
    from safetensors import safe_open
    mismatches = []
    with safe_open(st_path, framework='np') as f:
        st_names = f.keys()
        for name in st_names:
            st_arr = f.get_tensor(name)
            gg_arr = gguf['tensors'].get(name)
            if gg_arr is None:
                mismatches.append((name, 'missing in gguf'))
            elif not np.array_equal(gg_arr, st_arr):
                mismatches.append((name, 'bitwise mismatch'))
        for name in gguf['tensors']:
            if name not in st_names:
                mismatches.append((name, 'missing in safetensors'))
    return mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?', help='path to a .gguf file')
    ap.add_argument('--all', action='store_true', help='verify all gguf in artifacts')
    ap.add_argument('--artifacts', default=os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
    ap.add_argument('--check-safetensors', default=None,
                    help='compare bitwise against this .safetensors')
    args = ap.parse_args()

    if args.all:
        artifacts = os.path.abspath(args.artifacts)
        models = sorted(d for d in os.listdir(artifacts)
                        if os.path.isdir(os.path.join(artifacts, d)))
        failed = []
        for m in models:
            gguf = os.path.join(artifacts, m, f'{m}.gguf')
            st = os.path.join(artifacts, m, f'{m}.safetensors')
            g = load_gguf(gguf)
            mm = check_against_safetensors(gguf, st)
            if mm:
                failed.append(m)
                print(f'✗ {m}: {len(mm)} mismatches')
                for n, why in mm[:3]:
                    print(f'    {n}: {why}')
            else:
                print(f'✓ {m}: {g["tensor_count"]} tensors, '
                      f'{g["metadata"].get("general.name")}, '
                      f'alignment={g["alignment"]}, gguf v{g["version"]}')
        if failed:
            print(f'FAILED: {failed}')
            sys.exit(1)
        print('\nAll gguf files read and bitwise-verified against safetensors.')
        return

    if not args.path:
        ap.error('provide a path or --all')
    gguf = load_gguf(args.path)
    print(json.dumps({
        'version': gguf['version'],
        'metadata': gguf['metadata'],
        'tensor_count': gguf['tensor_count'],
        'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
                    for k, v in gguf['tensors'].items()},
    }, indent=2))
    if args.check_safetensors:
        mm = check_against_safetensors(args.path, args.check_safetensors)
        if mm:
            print('MISMATCHES:', mm)
            sys.exit(1)
        print('Bitwise-verified against', args.check_safetensors)


if __name__ == '__main__':
    main()
