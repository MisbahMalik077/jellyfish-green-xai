"""TNG50 jellyfish dataset builder.

Labels: Cosmological Jellyfish (Zinger et al. 2024) Zooniverse scores, TNG50-1 supplementary
catalogue `jellyfish.hdf5`. Images: rendered by us from gas/stellar particle cutouts fetched
from the IllustrisTNG web API (serial requests, as the TNG team asks).

Every catalogue entry belongs to a tracked "branch" (the same galaxy across snapshots). We keep the
branch index so that train/val/test splits are done by galaxy, never by image.
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
from scipy.ndimage import gaussian_filter

BASE = "https://www.tng-project.org/api/TNG50-1"
BOX = 35000.0  # ckpc/h, TNG50-1
NPIX = 128

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "jellyfish_TNG50-1.hdf5"
MAPS = ROOT / "data" / "processed" / "maps"
META = ROOT / "data" / "processed" / "metadata.jsonl"
SAMPLE = ROOT / "data" / "processed" / "sample.csv"


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    if "TNG_API_KEY" not in os.environ:
        raise RuntimeError("Set TNG_API_KEY (in .env or the environment).")
    return {"api-key": os.environ["TNG_API_KEY"]}


# --------------------------------------------------------------------------- sample selection
def build_catalogue(snap_min: int = 67, snap_max: int = 99) -> pd.DataFrame:
    """All inspected TNG50 satellites in [snap_min, snap_max] with score and branch index."""
    f = h5py.File(RAW, "r")
    B, S = f["Branches_SubfindID"][:], f["Branches_SnapNum"][:]
    key2branch = {}
    for b in range(B.shape[0]):
        for c in range(B.shape[1]):
            if B[b, c] >= 0:
                key2branch[(int(S[b, c]), int(B[b, c]))] = b
    rows = []
    for snap in range(snap_min, snap_max + 1):
        g = f.get(f"Snapshot_{snap:03d}")
        if g is None:
            continue
        for sid, sc in zip(g["SubhaloIDs"][:], g["ScoreAdjusted"][:]):
            rows.append(dict(snap=snap, subhalo_id=int(sid), score=float(sc),
                             branch=key2branch.get((snap, int(sid)), -1)))
    return pd.DataFrame(rows)


def select_sample(n_per_class: int = 1000, jf_thresh: float = 0.8, non_thresh: float = 0.2,
                  max_jf_per_branch: int = 2, min_snap_gap: int = 8, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat = build_catalogue()
    cat = cat[cat.branch >= 0]

    # jellyfish: up to `max_jf_per_branch` epochs per galaxy, separated by >= min_snap_gap snapshots
    jf = cat[cat.score >= jf_thresh]
    picks = []
    for _, grp in jf.groupby("branch"):
        grp = grp.sample(frac=1, random_state=int(rng.integers(1e9)))
        chosen = []
        for _, r in grp.iterrows():
            if all(abs(r.snap - c.snap) >= min_snap_gap for c in chosen):
                chosen.append(r)
            if len(chosen) == max_jf_per_branch:
                break
        picks += chosen
    jf_sel = pd.DataFrame(picks)
    if len(jf_sel) > n_per_class:
        jf_sel = jf_sel.sample(n_per_class, random_state=seed)

    # non-jellyfish: clean negatives = galaxies never scored >= 0.5 at any inspected epoch;
    # one epoch per galaxy.
    max_score = cat.groupby("branch").score.max()
    clean_branches = max_score[max_score < 0.5].index
    non = cat[(cat.score <= non_thresh) & cat.branch.isin(clean_branches)]
    non_sel = non.groupby("branch").sample(1, random_state=seed)
    non_sel = non_sel.sample(min(len(jf_sel), len(non_sel)), random_state=seed)

    jf_sel = jf_sel.assign(label=1)
    non_sel = non_sel.assign(label=0)
    sample = pd.concat([jf_sel, non_sel]).reset_index(drop=True)
    sample[["snap", "subhalo_id", "branch", "label"]] = sample[["snap", "subhalo_id", "branch", "label"]].astype(int)
    return sample


# --------------------------------------------------------------------------- fetch + render
def _get(url, headers, params=None, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=120)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(5 * (k + 1))
                continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"failed: {url}")


def _project(coords, masses, centre, half_width, npix=NPIX):
    d = coords - centre
    d = (d + BOX / 2) % BOX - BOX / 2  # periodic wrap
    m = (np.abs(d[:, 0]) < half_width) & (np.abs(d[:, 1]) < half_width)
    H, _, _ = np.histogram2d(d[m, 0], d[m, 1], bins=npix,
                             range=[[-half_width, half_width], [-half_width, half_width]],
                             weights=masses[m])
    pix_area = (2 * half_width / npix) ** 2
    return (H.T / pix_area).astype(np.float32)  # surface density, code units / (ckpc/h)^2


def render(cat: dict, cut: h5py.File, fov_factor: float = 15.0, fov_min: float = 20.0, fov_max: float = 80.0):
    """Project gas and stars along the simulation z-axis onto a 128x128 grid."""
    centre = np.array([cat["pos_x"], cat["pos_y"], cat["pos_z"]])
    rh = float(cat["halfmassrad_stars"])
    hw = float(np.clip(fov_factor * rh, fov_min, fov_max))
    out = {"half_width": hw, "rhalf_stars": rh}
    for name, pt in (("gas", "PartType0"), ("stars", "PartType4")):
        if pt in cut:
            m = _project(cut[pt]["Coordinates"][:], cut[pt]["Masses"][:], centre, hw)
        else:
            m = np.zeros((NPIX, NPIX), np.float32)
        out[name] = gaussian_filter(m, 1.0)
    return out


CAT_KEYS = ["mass_stars", "mass_gas", "sfr", "halfmassrad_stars", "halfmassrad_gas", "grnr",
            "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "len_gas", "len_stars"]


def fetch_all(sample: pd.DataFrame, log_every: int = 25):
    H = load_env()
    MAPS.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in MAPS.glob("*.npz")}
    t0, n = time.time(), 0
    with open(META, "a") as meta_f:
        for r in sample.itertuples(index=False):
            key = f"{r.snap:03d}_{r.subhalo_id}"
            if key in done:
                continue
            url = f"{BASE}/snapshots/{r.snap}/subhalos/{r.subhalo_id}/"
            cat = _get(url, H).json()
            cut = h5py.File(io.BytesIO(_get(url + "cutout.hdf5", H,
                                            {"gas": "Coordinates,Masses", "stars": "Coordinates,Masses"}).content), "r")
            img = render(cat, cut)
            np.savez_compressed(MAPS / f"{key}.npz", gas=img["gas"].astype(np.float16),
                                stars=img["stars"].astype(np.float16))
            rec = dict(key=key, snap=int(r.snap), subhalo_id=int(r.subhalo_id), branch=int(r.branch),
                       label=int(r.label), score=float(r.score), half_width=img["half_width"],
                       **{k: cat.get(k) for k in CAT_KEYS})
            meta_f.write(json.dumps(rec) + "\n")
            meta_f.flush()
            n += 1
            if n % log_every == 0:
                el = time.time() - t0
                print(f"{len(done) + n}/{len(sample)}  {el / n:.2f}s/galaxy", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    if SAMPLE.exists():
        s = pd.read_csv(SAMPLE)
    else:
        s = select_sample()
        s.to_csv(SAMPLE, index=False)
    print(s.label.value_counts().to_dict(), "branches:", s.branch.nunique(), flush=True)
    fetch_all(s)
