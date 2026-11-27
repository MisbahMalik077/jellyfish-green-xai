"""Dataset, normalisation, galaxy-level (branch) splits and physical masks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
MAPS = ROOT / "data" / "processed" / "maps"
META = ROOT / "data" / "processed" / "metadata.jsonl"

# log10 surface density (code units: 1e10 Msun/h per (ckpc/h)^2) display range, fixed for all images
LOG_MIN, LOG_MAX = -7.0, -2.0
GAS_THRESH = -5.5  # log10 Sigma_gas above which a pixel counts as "gas present" for the tail mask
BODY_RADIUS_FACTOR = 2.0  # stellar body = projected radius < 2 r_half,*


SAMPLE_MATCHED = ROOT / "data" / "processed" / "sample_matched.csv"


def load_metadata(matched: bool = True) -> pd.DataFrame:
    """Per-image metadata. matched=True restricts to the mass/gas-fraction/redshift-matched sample
    (src/match_negatives.py); labels and scores are taken from the sample file."""
    recs = [json.loads(l) for l in open(META)]
    df = pd.DataFrame(recs).drop_duplicates("key").reset_index(drop=True)
    df = df[[(MAPS / f"{k}.npz").exists() for k in df.key]]
    if matched:
        s = pd.read_csv(SAMPLE_MATCHED)
        s["key"] = [f"{a:03d}_{b}" for a, b in zip(s.snap, s.subhalo_id)]
        df = df.drop(columns=["label", "score", "branch"]).merge(s[["key", "label", "score", "branch"]], on="key")
    return df.reset_index(drop=True)


def make_splits(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """70/15/15 split stratified by label and grouped by galaxy branch (no galaxy in two splits)."""
    df = df.copy()
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold = np.zeros(len(df), int)
    for k, (_, idx) in enumerate(sgkf.split(df, df.label, df.branch)):
        fold[idx] = k
    df["split"] = np.where(fold < 14, "train", np.where(fold < 17, "val", "test"))
    assert not (set(df[df.split == "train"].branch) & set(df[df.split == "test"].branch))
    return df


def to_image(gas: np.ndarray) -> np.ndarray:
    x = np.log10(gas.astype(np.float32) + 1e-12)
    return np.clip((x - LOG_MIN) / (LOG_MAX - LOG_MIN), 0, 1).astype(np.float32)


def load_maps(key: str):
    d = np.load(MAPS / f"{key}.npz")
    return d["gas"].astype(np.float32), d["stars"].astype(np.float32)


def physical_masks(row, npix: int = 128):
    """Returns (body, gas, tail) boolean masks in image pixels.

    body: projected radius < 2 r_half,*;  gas: log Sigma_gas > GAS_THRESH;  tail: gas & ~body.
    """
    gas, _ = load_maps(row.key)
    pix = 2 * row.half_width / npix
    r_body = BODY_RADIUS_FACTOR * row.halfmassrad_stars / pix
    yy, xx = np.mgrid[:npix, :npix] - (npix - 1) / 2
    body = np.hypot(xx, yy) < r_body
    gasm = np.log10(gas + 1e-12) > GAS_THRESH
    return body, gasm, gasm & ~body


class GasMapDataset(Dataset):
    """Single-channel log gas surface-density maps. Augmentation = flips + 90 deg rotations
    (exact symmetries of a projected map, so labels are preserved)."""

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.x = np.stack([to_image(load_maps(k)[0]) for k in self.df.key])[:, None]
        self.y = self.df.label.values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        x = self.x[i]
        if self.augment:
            x = np.rot90(x, np.random.randint(4), axes=(1, 2))
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), self.y[i]
