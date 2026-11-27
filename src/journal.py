"""Journal-version experiments (resumable; every unit of work is saved under results/journal/).

  A  resolution test   Grad-CAM at an 8x8 mid-level layer + full-resolution, model-agnostic attributions
                       (Integrated Gradients, occlusion) for all models and seeds, scored against the physical masks.
  B  sanity check      model-parameter randomisation (Adebayo et al. 2018): similarity of maps before/after
                       re-initialising the classifier head and the whole network.
  C  hold-out tests    unseen galaxies at z ~ 0.5-2 (labels known) and ambiguous volunteer scores (0.2-0.8).
  D  physical drivers  explanation fidelity / recall versus host mass, clustercentric distance, tail area, score.
  E  learning curve    training-set fraction 10/25/50% (+100% from the main runs), measured energy.
  F  hardware          CPU-only inference energy on the same machine (portability of energy rankings).
  G  energy of A       measured energy per 1k explanations for the new attribution methods.

Stages that measure energy (E, F, G) wait until the background TNG download has finished, so the
network process does not contaminate the readings. Usage:  python journal.py [A B ...]
"""
from __future__ import annotations

import gc
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients, Occlusion
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.stats import spearmanr

import data as D
import train
import xai
from energy import PowerMeter, measure_idle
from models import MODELS, build, vit_reshape

warnings.filterwarnings("ignore")
ROOT = D.ROOT
OUT = ROOT / "results" / "journal"; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2]
DEV = train.DEVICE
MID = {"resnet50": lambda m: m.layer3[-1], "resnet18": lambda m: m.layer3[-1],
       "efficientnet_b0": lambda m: m.blocks[4], "mobilenetv3_small_100": lambda m: m.blocks[3]}
FETCH_LOG = ROOT / "logs" / "fetch_extra.log"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_model(name, seed, ckpt_dir=None):
    m = build(name, pretrained=False).to(DEV)
    p = (ckpt_dir or ROOT / "checkpoints") / f"{name}_s{seed}.pt"
    m.load_state_dict(torch.load(p, map_location=DEV)); return m.eval()


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    if DEV.type == "mps":
        torch.mps.empty_cache()


def test_set():
    df = pd.read_csv(ROOT / "data/processed/splits.csv")
    te = D.GasMapDataset(df[df.split == "test"])
    jf = te.df.label.values == 1
    return te, torch.from_numpy(te.x[jf]), list(te.df[jf].itertuples())


# ----------------------------------------------------------------------------- attribution methods
def gradcam_layer(model, name, layer, x, bs=32):
    cam = GradCAM(model=model, target_layers=[layer], reshape_transform=vit_reshape if name.startswith("vit") else None)
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(DEV)
        out.append(cam(input_tensor=xb, targets=[ClassifierOutputTarget(1)] * len(xb)))
    return np.concatenate(out).astype(np.float32)


def integrated_gradients(model, x, steps=32, bs=8):
    ig = IntegratedGradients(model)
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(DEV).requires_grad_(True)
        a = ig.attribute(xb, baselines=torch.zeros_like(xb), target=1, n_steps=steps, internal_batch_size=bs * 8)
        out.append(a.detach().clamp(min=0).sum(1).cpu().numpy())  # positive evidence for "jellyfish"
    return np.concatenate(out).astype(np.float32)


def occlusion(model, x, win=16, stride=8, bs=8):
    occ = Occlusion(model)
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(DEV)
        a = occ.attribute(xb, sliding_window_shapes=(1, win, win), strides=(1, stride, stride),
                          baselines=0.0, target=1, perturbations_per_eval=64)
        out.append(a.clamp(min=0).sum(1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def run_method(method, model, name, x):
    if method == "gradcam":
        return xai.compute_cams(model, name, "gradcam", x)
    if method == "gradcam_mid":
        return gradcam_layer(model, name, MID[name](model), x)
    if method == "ig":
        return integrated_gradients(model, x)
    if method == "occlusion":
        return occlusion(model, x)
    raise ValueError(method)


def new_methods(name):
    return ["gradcam"] + (["gradcam_mid"] if name in MID else []) + ["ig", "occlusion"]


def gas_lift(maps, rows):
    """Tail share of the attribution *within gas pixels*, relative to the tail share of gas pixels.
    Removes the advantage of methods that cannot attribute to empty sky (e.g. IG with a zero baseline)."""
    out = []
    for c, r in zip(maps, rows):
        _, gas, tail = D.physical_masks(r)
        cg = (c * gas).sum()
        out.append(((c * tail).sum() / cg) / (tail.sum() / gas.sum()) if cg > 0 and tail.sum() > 0 else np.nan)
    return out


def score_maps(maps, model, x, rows, probs, extra):
    d, _ = xai.deletion_auc(model, x, maps)
    tg = gas_lift(maps, rows)
    return [dict(**extra, key=r.key, p_jf=probs[i], DEL=d[i], TEL_gas=tg[i], **pm)
            for i, (pm, r) in enumerate(zip(xai.physical_metrics(maps, rows), rows))]


# ----------------------------------------------------------------------------- A
def stage_A():
    te, x, rows = test_set()
    jf = te.df.label.values == 1
    keep = {}
    for seed in SEEDS:
        for name in MODELS:
            f = OUT / f"A_{name}_s{seed}.csv"
            if f.exists():
                continue
            m = load_model(name, seed); probs = train.predict(m, te)[jf]; res = []
            for meth in new_methods(name):
                t0 = time.time(); c = run_method(meth, m, name, x)
                res += score_maps(c, m, x, rows, probs, dict(model=name, method=meth, seed=seed))
                if seed == 0:
                    keep[(name, meth)] = c[:12].astype(np.float16)
                log("A", name, seed, meth, f"{time.time() - t0:.0f}s")
                free(c)
            pd.DataFrame(res).to_csv(f, index=False); free(m)
    if keep:
        np.save(OUT / "A_maps_seed0.npy", keep, allow_pickle=True)


# ----------------------------------------------------------------------------- B
def _randomise(model, which):
    import copy
    r = copy.deepcopy(model)
    mods = list(r.modules())
    if which == "head":
        lin = [mm for mm in mods if isinstance(mm, torch.nn.Linear)][-1]
        lin.reset_parameters()
    else:
        torch.manual_seed(123)
        for mm in mods:
            if hasattr(mm, "reset_parameters") and mm is not r:
                mm.reset_parameters()
    return r.eval()


def stage_B():
    f = OUT / "B_sanity.csv"
    if f.exists():
        return
    te, x, rows = test_set()
    x = x[:64]; res = []
    for name in MODELS:
        m = load_model(name, 0)
        meths = {"gradcam": lambda mm: xai.compute_cams(mm, name, "gradcam", x), "ig": lambda mm: integrated_gradients(mm, x)}
        for mname, fn in meths.items():
            base = fn(m)
            for which in ("head", "full"):
                r = _randomise(m, which); c = fn(r)
                rho = [spearmanr(a.ravel(), b.ravel()).statistic for a, b in zip(base, c)]
                res.append(dict(model=name, method=mname, randomised=which, spearman=np.nanmean(np.abs(rho)),
                                TEL_orig=np.nanmean([p["TEL"] for p in xai.physical_metrics(base, rows[:64])]),
                                TEL_rand=np.nanmean([p["TEL"] for p in xai.physical_metrics(c, rows[:64])])))
                free(r, c)
            log("B", name, mname)
        free(m)
    pd.DataFrame(res).to_csv(f, index=False)


# ----------------------------------------------------------------------------- C
def extra_sets():
    meta = pd.DataFrame([json.loads(l) for l in open(ROOT / "data/processed/metadata_extra.jsonl")]).drop_duplicates("key")
    old = D.MAPS; D.MAPS = ROOT / "data/processed/maps_extra"
    try:
        ds = D.GasMapDataset(meta)
    finally:
        D.MAPS = old
    return meta, ds


def stage_C():
    f = OUT / "C_holdout_preds.csv"
    if f.exists():
        return
    meta, ds = extra_sets(); res = []
    hz_jf = (meta.subset.values == "highz") & (meta.label.values == 1)
    x_hz = torch.from_numpy(ds.x[hz_jf]); rows_hz = list(meta[hz_jf].itertuples())
    tel = []
    for name in MODELS:
        for seed in SEEDS:
            m = load_model(name, seed); p = train.predict(m, ds)
            res.append(pd.DataFrame(dict(model=name, seed=seed, key=meta.key.values, subset=meta.subset.values,
                                         label=meta.label.values, score=meta.score.values, p_jf=p)))
            old = D.MAPS; D.MAPS = ROOT / "data/processed/maps_extra"
            try:
                c = xai.compute_cams(m, name, "gradcam", x_hz)
                pm = xai.physical_metrics(c, rows_hz)
            finally:
                D.MAPS = old
            tel += [dict(model=name, seed=seed, key=r.key, p_jf=p[hz_jf][i], **q) for i, (q, r) in enumerate(zip(pm, rows_hz))]
            free(m, c)
        log("C", name)
    pd.concat(res).to_csv(f, index=False)
    pd.DataFrame(tel).to_csv(OUT / "C_highz_gradcam.csv", index=False)


# ----------------------------------------------------------------------------- energy stages
def wait_for_fetch():
    while FETCH_LOG.exists() and "ALL DONE" not in FETCH_LOG.read_text() and "Traceback" not in FETCH_LOG.read_text():
        log("waiting for TNG download to finish before energy measurements ..."); time.sleep(120)


def stage_G():
    """Energy per 1k explanations for the new methods (seed 0)."""
    f = OUT / "G_cost.csv"
    if f.exists():
        return
    wait_for_fetch(); measure_idle(15)
    te, x, rows = test_set(); res = []
    for name in MODELS:
        m = load_model(name, 0)
        for meth in [mm for mm in new_methods(name) if mm != "gradcam"]:
            run_method(meth, m, name, x[:8]); train._sync()
            with PowerMeter() as pm:
                run_method(meth, m, name, x); train._sync()
            r = pm.result("x_")
            res.append(dict(model=name, method=meth, per1k_joules=r["x_joules"] * 1000 / len(x),
                            per1k_seconds=r["x_seconds"] * 1000 / len(x), mode=r["x_energy_mode"]))
            log("G", name, meth, f"{res[-1]['per1k_joules']:.0f} J/1k")
        free(m)
    pd.DataFrame(res).to_csv(f, index=False)


def stage_F():
    """Inference energy with the CPU only (same laptop), seed-0 checkpoints."""
    f = OUT / "F_cpu_inference.csv"
    if f.exists():
        return
    wait_for_fetch(); measure_idle(15)
    te, _, _ = test_set(); res = []
    cpu = torch.device("cpu"); old = train.DEVICE
    train.DEVICE = cpu
    try:
        for name in MODELS:
            m = build(name, pretrained=False)
            m.load_state_dict(torch.load(ROOT / f"checkpoints/{name}_s0.pt", map_location=cpu)); m.eval()
            train.predict(m, te)
            n, t0 = 0, time.time()
            with PowerMeter() as pm:
                while time.time() - t0 < 20:
                    train.predict(m, te); n += len(te)
            r = pm.result("i_")
            res.append(dict(model=name, device="cpu", per1k_joules=r["i_joules"] * 1000 / n,
                            per1k_seconds=r["i_seconds"] * 1000 / n, cpu_frac=r.get("i_cpu_joules", 0) / r["i_joules"]))
            log("F", name, f"{res[-1]['per1k_joules']:.1f} J/1k")
            free(m)
    finally:
        train.DEVICE = old
    pd.DataFrame(res).to_csv(f, index=False)


LC_MODELS = ["resnet50", "resnet18", "mobilenetv3_small_100", "vit_tiny_patch16_224"]
LC_FRACS = [0.1, 0.25, 0.5]


def stage_E():
    wait_for_fetch(); measure_idle(15)
    df = pd.read_csv(ROOT / "data/processed/splits.csv")
    va, te = D.GasMapDataset(df[df.split == "val"]), D.GasMapDataset(df[df.split == "test"])
    jf = te.df.label.values == 1; x = torch.from_numpy(te.x[jf]); rows = list(te.df[jf].itertuples())
    trdf = df[df.split == "train"]
    old_runs, old_ck = train.RUNS, train.CKPT
    for frac in LC_FRACS:
        # subsample whole galaxies (branches), stratified by label, nested across fractions
        sub = []
        for lab, g in trdf.groupby("label"):
            br = np.random.default_rng(0).permutation(g.branch.unique())
            keep = set(br[:max(1, int(round(frac * len(br))))])
            sub.append(g[g.branch.isin(keep)])
        tr = D.GasMapDataset(pd.concat(sub), augment=True)
        train.RUNS = OUT / f"E_runs_f{frac}"; train.CKPT = ROOT / "checkpoints" / f"lc_f{frac}"
        for name in LC_MODELS:
            for seed in SEEDS:
                fout = train.RUNS / f"{name}_s{seed}.json"
                if fout.exists() and "TEL_gradcam" in json.loads(fout.read_text()):
                    continue
                r = train.train_one(name, seed, tr, va, te)
                m = load_model(name, seed, train.CKPT)
                c = xai.compute_cams(m, name, "gradcam", x)
                r["TEL_gradcam"] = float(np.nanmean([p["TEL"] for p in xai.physical_metrics(c, rows)]))
                r["n_train"] = len(tr); r["frac"] = frac
                fout.write_text(json.dumps(r)); free(m, c)
                log("E", frac, name, seed, f"acc={r['accuracy']:.3f} TEL={r['TEL_gradcam']:.2f} E={r['train_joules']:.0f}J")
    train.RUNS, train.CKPT = old_runs, old_ck


# ----------------------------------------------------------------------------- H
def stage_H():
    """Reference-free agreement: Spearman correlation of each cheap map with the occlusion map of the same
    galaxy (whole image and gas pixels only), all models and seeds."""
    f = OUT / "H_agreement_occlusion.csv"
    if f.exists():
        return
    te, x, rows = test_set(); res = []
    gas = [D.physical_masks(r)[1] for r in rows]
    for seed in SEEDS:
        for name in MODELS:
            m = load_model(name, seed)
            occ = occlusion(m, x)
            for meth in ["gradcam", "gradcam++", "ig"]:
                c = integrated_gradients(m, x) if meth == "ig" else xai.compute_cams(m, name, meth, x)
                for i, r in enumerate(rows):
                    g = gas[i]
                    res.append(dict(model=name, seed=seed, method=meth, key=r.key,
                                    rho_all=spearmanr(c[i].ravel(), occ[i].ravel()).statistic,
                                    rho_gas=spearmanr(c[i][g], occ[i][g]).statistic if g.sum() > 10 else np.nan))
                free(c)
            free(m, occ); log("H", name, seed)
    pd.DataFrame(res).to_csv(f, index=False)


if __name__ == "__main__":
    stages = sys.argv[1:] or ["A", "B", "G", "F", "E", "C"]
    for s in stages:
        log("=== stage", s); globals()[f"stage_{s}"](); log("=== done", s)
    log("JOURNAL ALL DONE")
