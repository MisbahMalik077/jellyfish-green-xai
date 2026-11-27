"""Matched negative sample.

Problem found in the first sample: 'clean' non-jellyfish were far more gas-rich than jellyfish (median
M_gas/M_* 1.86 vs 0.30), so a classifier could separate the classes from total gas content alone.
Fix: for every jellyfish we pick the nearest clean non-jellyfish (without replacement, at most one epoch
per galaxy) in (log M_*, log M_gas/M_*, snapshot). Only gas *morphology* then distinguishes the classes.

Steps (resumable): 1) fetch catalogue values for candidate negatives; 2) match; 3) fetch + render the
new negatives' cutouts (reusing maps already on disk).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import tng_data as T

CAND = T.ROOT / "data" / "processed" / "neg_candidates.jsonl"
SAMPLE_M = T.ROOT / "data" / "processed" / "sample_matched.csv"
EPOCHS_PER_BRANCH = 3


def candidates(seed=0):
    cat = T.build_catalogue()
    cat = cat[cat.branch >= 0]
    mx = cat.groupby("branch").score.max()
    clean = cat[(cat.score <= 0.2) & cat.branch.isin(mx[mx < 0.5].index)]
    return (clean.sample(frac=1, random_state=seed).groupby("branch").head(EPOCHS_PER_BRANCH)
            .reset_index(drop=True))


def fetch_candidate_props(cands):
    H = T.load_env()
    done = set()
    if CAND.exists():
        done = {json.loads(l)["key"] for l in open(CAND)}
    t0, n = time.time(), 0
    with open(CAND, "a") as f:
        for r in cands.itertuples(index=False):
            key = f"{r.snap:03d}_{r.subhalo_id}"
            if key in done:
                continue
            c = T._get(f"{T.BASE}/snapshots/{r.snap}/subhalos/{r.subhalo_id}/", H).json()
            f.write(json.dumps(dict(key=key, snap=int(r.snap), subhalo_id=int(r.subhalo_id), branch=int(r.branch),
                                    score=float(r.score), mass_stars=c["mass_stars"], mass_gas=c["mass_gas"])) + "\n")
            f.flush()
            n += 1
            if n % 200 == 0:
                print(f"candidates {len(done) + n}/{len(cands)}  {(time.time() - t0) / n:.2f}s each", flush=True)


def feats(df):
    ms = np.log10(np.clip(df.mass_stars.values, 1e-6, None))
    fg = np.log10(np.clip(df.mass_gas.values / np.clip(df.mass_stars.values, 1e-6, None), 1e-3, None))
    return np.c_[ms / 0.5, fg / 0.3, df.snap.values / 8.0]  # scales ~ tolerance per feature


def match():
    meta = pd.DataFrame([json.loads(l) for l in open(T.META)]).drop_duplicates("key")
    jf = meta[meta.label == 1].copy()
    gas_pix = []
    for k in jf.key:  # drop jellyfish whose bound-gas map is empty
        g = np.load(T.MAPS / f"{k}.npz")["gas"].astype(np.float32)
        gas_pix.append((np.log10(g + 1e-12) > -5.5).sum())
    jf = jf[np.array(gas_pix) > 0]
    cand = pd.DataFrame([json.loads(l) for l in open(CAND)]).drop_duplicates("key")
    cand = cand[cand.mass_gas > 0].reset_index(drop=True)
    Fj, Fc = feats(jf), feats(cand)
    rng = np.random.default_rng(0)
    used_rows, used_branch, pairs = set(), set(), []
    for i in rng.permutation(len(jf)):
        d = np.linalg.norm(Fc - Fj[i], axis=1)
        for j in np.argsort(d):
            if j not in used_rows and cand.branch[j] not in used_branch:
                used_rows.add(j); used_branch.add(cand.branch[j]); pairs.append((i, j, d[j]))
                break
    neg = cand.loc[[j for _, j, _ in pairs]].assign(label=0)
    print(f"matched {len(pairs)} pairs; median distance {np.median([p[2] for p in pairs]):.2f}", flush=True)
    s = pd.concat([jf[["snap", "subhalo_id", "branch", "label", "score"]],
                   neg[["snap", "subhalo_id", "branch", "label", "score"]]]).reset_index(drop=True)
    s[["snap", "subhalo_id", "branch", "label"]] = s[["snap", "subhalo_id", "branch", "label"]].astype(int)
    s.to_csv(SAMPLE_M, index=False)
    return s


if __name__ == "__main__":
    c = candidates()
    print("candidate negatives:", len(c), "from", c.branch.nunique(), "galaxies", flush=True)
    fetch_candidate_props(c)
    s = match()
    T.fetch_all(s)
