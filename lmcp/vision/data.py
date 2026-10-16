"""CIFAR-10 loading, augmentation and input deficits.

Downloading is NOT done by training code (program.md: no network calls from training). Run once:
    .venv/bin/python -m lmcp.vision.data
"""
import hashlib
import pickle
import tarfile
import urllib.request

import numpy as np

from .. import ROOT

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"   # official; was ~30 KB/s on 2026-10-16
MIRROR = "https://dl.fbaipublicfiles.com/cifar-10-python.tar.gz"      # byte-identical (md5 checked below)
MD5 = "c58f30108f718f92721af3b95e74349a"
DIR = ROOT / "data" / "cifar10"
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
KINDS = {"clean", "blur", "label_permute", "vflip"}


def download():
    DIR.mkdir(parents=True, exist_ok=True)
    tgz = DIR / "cifar-10-python.tar.gz"
    if not tgz.exists():
        for u in (MIRROR, URL):
            print("downloading", u, flush=True)
            try:
                urllib.request.urlretrieve(u, tgz)
                break
            except OSError as e:
                print("  failed:", e, flush=True)
                tgz.unlink(missing_ok=True)
    md5 = hashlib.md5(tgz.read_bytes()).hexdigest()
    if md5 != MD5:
        raise SystemExit(f"md5 mismatch for {tgz}: {md5} != {MD5}")
    with tarfile.open(tgz) as t:
        def load(name):
            d = pickle.load(t.extractfile(f"cifar-10-batches-py/{name}"), encoding="bytes")
            return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[b"labels"], dtype=np.int64)
        parts = [load(f"data_batch_{i}") for i in range(1, 6)]
        np.save(DIR / "train_x.npy", np.concatenate([p[0] for p in parts]).astype(np.uint8))
        np.save(DIR / "train_y.npy", np.concatenate([p[1] for p in parts]))
        x, y = load("test_batch")
        np.save(DIR / "test_x.npy", x.astype(np.uint8))
        np.save(DIR / "test_y.npy", y)
    print("cifar10 ready in", DIR)


def load_cifar():
    need = [DIR / f"{s}_{a}.npy" for s in ("train", "test") for a in "xy"]
    if not all(p.exists() for p in need):
        raise SystemExit("CIFAR-10 missing; run: .venv/bin/python -m lmcp.vision.data")
    return tuple(np.load(p) for p in need)  # train_x, train_y, test_x, test_y  (uint8 NHWC, int64)


def _upsample_matrix(n_out=32, n_in=8):
    """Bilinear interpolation matrix, half-pixel centres, edge clamped (matches align_corners=False)."""
    src = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
    lo = np.clip(np.floor(src).astype(int), 0, n_in - 1)
    hi = np.clip(lo + 1, 0, n_in - 1)
    w = np.clip(src - lo, 0.0, 1.0)
    U = np.zeros((n_out, n_in), dtype=np.float32)
    U[np.arange(n_out), lo] += 1 - w
    U[np.arange(n_out), hi] += w
    return U


def blur(x_uint8, size=8):
    """Paper's cataract: area-downsample 32x32 -> 8x8, bilinear-upsample back to 32x32. uint8 NHWC in/out."""
    n = x_uint8.shape[0]
    f = 32 // size
    small = x_uint8.astype(np.float32).reshape(n, size, f, size, f, 3).mean(axis=(2, 4))
    U = _upsample_matrix(32, size)
    out = np.einsum("ia,nabc,jb->nijc", U, small, U)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def vflip(x_uint8):
    return x_uint8[:, ::-1]


def label_perm(seed):
    """Fixed random derangement of the 10 classes (no label maps to itself)."""
    rng = np.random.default_rng([seed, 4242])
    while True:
        p = rng.permutation(10)
        if not (p == np.arange(10)).any():
            return p


def normalise(x_uint8):
    return ((x_uint8.astype(np.float32) / 255.0 - MEAN) / STD).astype(np.float32)


def augment(x_uint8, rng, pad=4):
    """Random crop (4px zero padding) + horizontal flip on a uint8 NHWC batch."""
    n = x_uint8.shape[0]
    xp = np.pad(x_uint8, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    ox, oy = rng.integers(0, 2 * pad + 1, size=(2, n))
    out = np.empty_like(x_uint8)
    flip = rng.random(n) < 0.5
    for i in range(n):
        c = xp[i, oy[i]:oy[i] + 32, ox[i]:ox[i] + 32]
        out[i] = c[:, ::-1] if flip[i] else c
    return out


if __name__ == "__main__":
    download()
