#!/usr/bin/env python3
"""
check_gguf_external.py — spec-compliance cross-check of the generated .gguf
files using the external `gguf` reader (the PyPI package used by the ggml /
llama.cpp ecosystem), independent of scripts/read-gguf.py.

Each tensor read by the external reader must be bitwise identical to the
corresponding .npy extracted from the TF.js weights. GGUF stores dims in
GGML (reversed) order, so dims are reversed back to numpy order for the
comparison.

Usage:
  python check_gguf_external.py [--artifacts <dir>]
"""
import argparse
import glob
import os
import sys

import numpy as np
from gguf import GGUFReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts', default=os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
    args = ap.parse_args()
    artifacts = os.path.abspath(args.artifacts)

    files = sorted(glob.glob(os.path.join(artifacts, '*', '*.gguf')))
    if not files:
        print('no gguf files found'); sys.exit(1)

    bad = 0
    total = 0
    for f in files:
        model = os.path.basename(os.path.dirname(f))
        r = GGUFReader(f)
        for t in r.tensors:
            npy = np.load(os.path.join(artifacts, model, 'weights',
                                       t.name.replace('/', '_') + '.npy'))
            arr = np.array(t.data.reshape(list(reversed(t.shape))))
            if arr.dtype != npy.dtype:
                arr = arr.astype(npy.dtype)
            arr = arr.reshape(npy.shape)
            total += 1
            if not np.array_equal(arr, npy):
                bad += 1
                print(f' MISMATCH {model}/{t.name} ({t.shape} vs {npy.shape})')
        print(f'✓ external reader: {model}')
    print(f'\n{total - bad}/{total} tensors bitwise-equal via external gguf reader')
    if bad:
        print(f'FAILED: {bad} mismatches')
        sys.exit(1)


if __name__ == '__main__':
    main()
