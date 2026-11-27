"""Extra data for the journal version (resumable, serial API requests as the TNG team asks).

1. high-z hold-out: jellyfish at snapshots 33-66 (z ~ 0.5-2) + mass/gas-fraction/snapshot-matched clean negatives.
2. intermediate-score hold-out: satellites with 0.2 < score < 0.8 at snapshots 67-99 (ambiguous to volunteers).
3. host properties (M200c, R200c, clustercentric distance) for every galaxy in all samples.

No branch (galaxy) that appears in the training sample (sample_matched.csv) is used in 1 or 2.
Maps go to data/processed/maps_extra/, metadata to data/processed/metadata_extra.jsonl.
"""
from __future__ import annotations

import io
import json
import time

import h5py
import numpy as np
import pandas as pd

import tng_data as T
from match_negatives import feats

P = T.ROOT / "data" / "processed"
MAPS_X = P / "maps_extra"
META_X = P / "metadata_extra.jsonl"
CAND_X = P / "neg_candidates_highz.jsonl"
SAMPLE_X = P / "sample_extra.csv"
HOSTS = P / "hosts.jsonl"
SNAP_HZ = (33, 66)


def get_json(url, H, tries=6):
    """GET + JSON decode with retries (the API occasionally returns an HTML error page)."""
    for k in range(tries):
        try:
            return T._get(url, H).json()
        except Exception as e:  # noqa: BLE001
            print("retry", url, type(e).__name__, flush=True); time.sleep(10 * (k + 1))
    return None


def _jsonl(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []


def train_branches():
    return set(pd.read_csv(P / "sample_matched.csv").branch)


def highz_jellyfish(seed=42):
    rng = np.random.default_rng(seed)
    cat = T.build_catalogue(*SNAP_HZ)
    cat = cat[(cat.branch >= 0) & ~cat.branch.isin(train_branches())]
    jf, picks = cat[cat.score >= 0.8], []
    for _, grp in jf.groupby("branch"):
        chosen = []
        for _, r in grp.sample(frac=1, random_state=int(rng.integers(1e9))).iterrows():
            if all(abs(r.snap - c.snap) >= 8 for c in chosen):
                chosen.append(r)
            if len(chosen) == 2:
                break
        picks += chosen
    jf_df = pd.DataFrame(picks).astype({"snap": int, "subhalo_id": int, "branch": int})
    return jf_df.assign(label=1, subset="highz"), cat


def highz_candidates(cat, n=900, seed=0):
    mx = cat.groupby("branch").score.max()
    clean = cat[(cat.score <= 0.2) & cat.branch.isin(mx[mx < 0.5].index)]
    clean = clean.sample(frac=1, random_state=seed).groupby("branch").head(2)
    return clean.sample(min(n, len(clean)), random_state=seed)


def intermediate(n_per_bin=100, seed=7):
    cat = T.build_catalogue(67, 99)
    cat = cat[(cat.branch >= 0) & ~cat.branch.isin(train_branches())]
    mid = cat[(cat.score > 0.2) & (cat.score < 0.8)]
    mid = mid.sample(frac=1, random_state=seed).groupby("branch").head(1)
    mid["bin"] = pd.cut(mid.score, np.linspace(0.2, 0.8, 7))
    out = mid.groupby("bin", observed=True).apply(lambda g: g.sample(min(n_per_bin, len(g)), random_state=seed))
    return out.reset_index(drop=True).drop(columns="bin", errors="ignore").assign(label=-1, subset="intermediate")


def fetch_props(cands, H):
    done = {r["key"] for r in _jsonl(CAND_X)}
    with open(CAND_X, "a") as f:
        for i, r in enumerate(cands.itertuples(index=False)):
            key = f"{r.snap:03d}_{r.subhalo_id}"
            if key in done:
                continue
            c = T._get(f"{T.BASE}/snapshots/{r.snap}/subhalos/{r.subhalo_id}/", H).json()
            f.write(json.dumps(dict(key=key, snap=int(r.snap), subhalo_id=int(r.subhalo_id), branch=int(r.branch),
                                    score=float(r.score), mass_stars=c["mass_stars"], mass_gas=c["mass_gas"])) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"highz candidate props {i}/{len(cands)}", flush=True)


def match_highz(jf_rows, H):
    """Fetch catalogue values of the high-z jellyfish, then nearest-neighbour match as in match_negatives."""
    cache = P / "highz_jf_props.jsonl"
    done = {(r["snap"], r["subhalo_id"]) for r in _jsonl(cache)}
    with open(cache, "a") as f:
        for r in jf_rows.itertuples(index=False):
            if (int(r.snap), int(r.subhalo_id)) in done:
                continue
            c = get_json(f"{T.BASE}/snapshots/{r.snap}/subhalos/{r.subhalo_id}/", H)
            if c is None:
                continue
            f.write(json.dumps(dict(snap=int(r.snap), subhalo_id=int(r.subhalo_id), branch=int(r.branch),
                                    score=float(r.score), mass_stars=c["mass_stars"], mass_gas=c["mass_gas"])) + "\n")
            f.flush()
    jf = pd.DataFrame(_jsonl(cache)).drop_duplicates(["snap", "subhalo_id"])
    jf = jf[jf.mass_gas > 0].reset_index(drop=True)
    cand = pd.DataFrame(_jsonl(CAND_X)).drop_duplicates("key")
    cand = cand[cand.mass_gas > 0].reset_index(drop=True)
    Fj, Fc = feats(jf), feats(cand)
    used_rows, used_branch, negs = set(), set(), []
    for i in np.random.default_rng(0).permutation(len(jf)):
        d = np.linalg.norm(Fc - Fj[i], axis=1)
        for j in np.argsort(d):
            if j not in used_rows and cand.branch[j] not in used_branch:
                used_rows.add(j); used_branch.add(cand.branch[j]); negs.append(j)
                break
    neg = cand.loc[negs]
    cols = ["snap", "subhalo_id", "branch", "score"]
    return pd.concat([jf[cols].assign(label=1), neg[cols].assign(label=0)]).assign(subset="highz")


def fetch_maps(sample, H):
    MAPS_X.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in MAPS_X.glob("*.npz")}
    t0, n = time.time(), 0
    with open(META_X, "a") as meta_f:
        for r in sample.itertuples(index=False):
            key = f"{r.snap:03d}_{r.subhalo_id}"
            if key in done:
                continue
            url = f"{T.BASE}/snapshots/{r.snap}/subhalos/{r.subhalo_id}/"
            cat = get_json(url, H)
            if cat is None:
                continue
            try:
                cut = h5py.File(io.BytesIO(T._get(url + "cutout.hdf5", H,
                                                  {"gas": "Coordinates,Masses", "stars": "Coordinates,Masses"}).content), "r")
            except Exception as e:  # noqa: BLE001
                print("skip cutout", key, type(e).__name__, flush=True); continue
            img = T.render(cat, cut)
            np.savez_compressed(MAPS_X / f"{key}.npz", gas=img["gas"].astype(np.float16),
                                stars=img["stars"].astype(np.float16))
            meta_f.write(json.dumps(dict(key=key, snap=int(r.snap), subhalo_id=int(r.subhalo_id), branch=int(r.branch),
                                         label=int(r.label), score=float(r.score), subset=r.subset,
                                         half_width=img["half_width"], **{k: cat.get(k) for k in T.CAT_KEYS})) + "\n")
            meta_f.flush()
            done.add(key); n += 1
            if n % 25 == 0:
                print(f"maps {len(done)}/{len(sample)}  {(time.time() - t0) / n:.2f}s/galaxy", flush=True)


def fetch_hosts(H):
    """Host-halo M200c, R200c and position for every (snap, grnr) in all metadata files."""
    meta = pd.DataFrame(_jsonl(T.META) + _jsonl(META_X)).drop_duplicates("key")
    done = {(h["snap"], h["grnr"]) for h in _jsonl(HOSTS)}
    todo = sorted({(int(s), int(g)) for s, g in zip(meta.snap, meta.grnr)} - done)
    print("host halos to fetch:", len(todo), flush=True)
    with open(HOSTS, "a") as f:
        for i, (s, g) in enumerate(todo):
            info = get_json(f"{T.BASE}/snapshots/{s}/halos/{g}/info.json", H)
            if info is None:
                continue
            f.write(json.dumps(dict(snap=s, grnr=g, m200c=info["Group_M_Crit200"], r200c=info["Group_R_Crit200"],
                                    gx=info["GroupPos"][0], gy=info["GroupPos"][1], gz=info["GroupPos"][2])) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"hosts {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    H = T.load_env()
    if SAMPLE_X.exists():
        sample = pd.read_csv(SAMPLE_X)
    else:
        jf_rows, cat_hz = highz_jellyfish()
        print("high-z jellyfish (unseen galaxies):", len(jf_rows), flush=True)
        fetch_props(highz_candidates(cat_hz), H)
        hz = match_highz(jf_rows, H)
        mid = intermediate()
        sample = pd.concat([hz, mid]).reset_index(drop=True)
        sample[["snap", "subhalo_id", "branch", "label"]] = sample[["snap", "subhalo_id", "branch", "label"]].astype(int)
        sample.to_csv(SAMPLE_X, index=False)
    print(sample.groupby(["subset", "label"]).size().to_dict(), flush=True)
    fetch_maps(sample, H)
    fetch_hosts(H)
    print("ALL DONE", flush=True)
