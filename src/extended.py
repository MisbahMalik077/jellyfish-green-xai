"""Extended analyses for the paper (no new training): shortcut baselines, full explanation table,
per-galaxy breakdowns and energy-component breakdown."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import data as D
from models import MODELS, PRETTY

OUT = D.ROOT / "results"
FIG = OUT / "figures"; TAB = OUT / "tables"
METHODS = ["gradcam", "gradcam++", "scorecam"]


def load_original():
    """First (unmatched) sample: 1000 jellyfish + 1000 'clean' negatives (data/processed/sample.csv)."""
    s = pd.read_csv(D.ROOT / "data" / "processed" / "sample.csv")
    keys = {f"{a:03d}_{b}" for a, b in zip(s.snap, s.subhalo_id)}
    df = D.load_metadata(matched=False)
    return df[df.key.isin(keys)].reset_index(drop=True)


def scalar_features(df):
    gp, gm = [], []
    for k in df.key:
        g, _ = D.load_maps(k)
        lg = np.log10(g + 1e-12)
        gp.append((lg > D.GAS_THRESH).mean()); gm.append(np.log10(g.sum() + 1e-12))
    ms = np.log10(df.mass_stars.clip(lower=1e-6))
    f = pd.DataFrame(dict(log_ms=ms, log_fg=np.log10((df.mass_gas / df.mass_stars.clip(lower=1e-6)).clip(lower=1e-3)),
                          snap=df.snap, log_ssfr=np.log10((df.sfr / df.mass_stars.clip(lower=1e-6)).clip(lower=1e-6)),
                          rgas_rstar=np.log10((df.halfmassrad_gas / df.halfmassrad_stars).clip(lower=1e-3)),
                          gas_area=gp, log_gas_map=gm))
    return f


SETS = {"Catalogue: $M_\\star$, $f_{\\rm gas}$, $z$": ["log_ms", "log_fg", "snap"],
        "Catalogue + sSFR, $r_{\\rm gas}/r_\\star$": ["log_ms", "log_fg", "snap", "log_ssfr", "rgas_rstar"],
        "Image: gas area, total gas": ["gas_area", "log_gas_map"],
        "All seven scalars": ["log_ms", "log_fg", "snap", "log_ssfr", "rgas_rstar", "gas_area", "log_gas_map"]}


def shortcut_baselines():
    rows = []
    for matched in (False, True):
        df = D.load_metadata(True) if matched else load_original()
        F = scalar_features(df)
        for name, cols in SETS.items():
            accs = {"LR": [], "GBT": []}
            for seed in range(3):
                s = D.make_splits(df, seed)
                tr, te = (s.split != "test").values, (s.split == "test").values
                for clf_name, clf in (("LR", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                                      ("GBT", HistGradientBoostingClassifier(max_depth=3, random_state=seed))):
                    clf.fit(F[cols][tr], df.label[tr])
                    p = clf.predict(F[cols][te])
                    # balanced accuracy
                    y = df.label[te].values
                    accs[clf_name].append(0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean()))
            rows.append(dict(sample="matched" if matched else "unmatched", features=name,
                             LR=np.mean(accs["LR"]), LR_sd=np.std(accs["LR"]),
                             GBT=np.mean(accs["GBT"]), GBT_sd=np.std(accs["GBT"])))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "shortcut_baselines.csv", index=False)
    return r


def boot_ci(per_gal, n=1000, seed=0):
    rng = np.random.default_rng(seed); v = per_gal.values
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def full_xai_table(x):
    rows = []
    for m in MODELS:
        for meth in METHODS:
            d = x[(x.model == m) & (x.method == meth)]
            g = d.groupby("key")[["TEL", "TEF", "GEF", "PG", "DEL"]].mean()
            seed_tel = d.groupby("seed").TEL.mean()
            lo, hi = boot_ci(g.DEL)
            rows.append(dict(model=m, method=meth, TEF=g.TEF.mean(), GEF=g.GEF.mean(), TEL=g.TEL.mean(),
                             TEL_seed_sd=seed_tel.std(ddof=0), DEL=g.DEL.mean(), DEL_lo=lo, DEL_hi=hi))
    return pd.DataFrame(rows)


def per_galaxy(x, meta):
    gc = x[x.method == "gradcam"].merge(meta[["key", "score"]], on="key")
    gc["correct"] = gc.p_jf >= 0.5
    gc["area_bin"] = pd.qcut(gc.tail_area, 3, labels=["small", "medium", "large"])
    gc["score_bin"] = pd.cut(gc.score, [0.79, 0.9, 0.97, 1.01], labels=["0.80-0.90", "0.90-0.97", ">0.97"])
    area = gc.groupby(["model", "area_bin"], observed=True).TEL.mean().unstack()
    score_rec = gc.groupby(["model", "score_bin"], observed=True).correct.mean().unstack()
    corr = gc.groupby(["model", "correct"]).TEL.mean().unstack()
    nwrong = gc[~gc.correct].groupby("model").size() / 3
    edges = gc.groupby("area_bin", observed=True).tail_area.agg(["min", "max"])
    nscore = gc[gc.model == "resnet50"].groupby("score_bin", observed=True).size() / 3
    return gc, area, score_rec, corr, nwrong, edges, nscore


def fig_per_galaxy(area, score_rec, nscore):
    cols = {m: c for m, c in zip(MODELS, plt.cm.tab10.colors)}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    xs = np.arange(3)
    for m in MODELS:
        ax[0].plot(xs, area.loc[m].values, "o-", color=cols[m], label=PRETTY[m])
        ax[1].plot(xs, score_rec.loc[m].values, "o-", color=cols[m], label=PRETTY[m])
    ax[0].axhline(1, ls="--", c="k", lw=0.8)
    ax[0].set_xticks(xs, ["small tail", "medium tail", "large tail"]); ax[0].set_ylabel("Grad-CAM tail lift (TEL)")
    ax[0].set_yscale("log"); ax[0].set_yticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
    ax[1].set_xticks(xs, [f"{b}\n(n={int(n)})" for b, n in zip(score_rec.columns, nscore.values)])
    ax[1].set_xlabel("volunteer score"); ax[1].set_ylabel("recall on test jellyfish")
    ax[1].legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig_per_galaxy.pdf", bbox_inches="tight", metadata={"Creator": None, "Producer": None})


def fig_data_check():
    un, ma = load_original(), D.load_metadata(True)
    Fu, Fm = scalar_features(un), scalar_features(ma)
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
    bins_fg = np.linspace(-3, 1.5, 31)
    for a, (df, F, ttl) in zip(ax[:2], ((un, Fu, "before matching"), (ma, Fm, "after matching"))):
        for lab, c, n in ((1, "tab:red", "jellyfish"), (0, "tab:blue", "non-jellyfish")):
            a.hist(F.log_fg[df.label == lab], bins_fg, histtype="step", density=True, color=c, lw=1.4, label=n)
        a.set_title(ttl, fontsize=10); a.set_xlabel(r"$\log\,M_{\rm gas}/M_\star$")
    ax[0].set_ylabel("density"); ax[0].legend(fontsize=8)
    bins_a = np.linspace(0, 1, 31)
    for lab, c in ((1, "tab:red"), (0, "tab:blue")):
        ax[2].hist(Fm.gas_area[ma.label == lab], bins_a, histtype="step", density=True, color=c, lw=1.4)
    ax[2].set_yscale("log"); ax[2].set_title("after matching", fontsize=10)
    ax[2].set_xlabel("fraction of image covered by gas")
    fig.tight_layout(); fig.savefig(FIG / "fig_data_check.pdf", bbox_inches="tight", metadata={"Creator": None, "Producer": None})
    return dict(fg_med_un=(Fu.log_fg[un.label == 1].median(), Fu.log_fg[un.label == 0].median()),
                fg_med_ma=(Fm.log_fg[ma.label == 1].median(), Fm.log_fg[ma.label == 0].median()),
                area_med_ma=(Fm.gas_area[ma.label == 1].median(), Fm.gas_area[ma.label == 0].median()),
                area_full_frac=((Fm.gas_area[ma.label == 1] > 0.5).mean(), (Fm.gas_area[ma.label == 0] > 0.5).mean()))


def energy_breakdown():
    r = pd.read_csv(OUT / "runs.csv")
    comp = r.groupby("model")[["train_cpu_joules", "train_gpu_joules", "train_dram_joules", "train_ane_joules",
                              "train_joules", "train_mean_watts", "epochs", "train_seconds"]].mean().loc[MODELS]
    comp["j_per_epoch"] = comp.train_joules / comp.epochs
    tot = comp[["train_cpu_joules", "train_gpu_joules", "train_dram_joules", "train_ane_joules"]].sum(1)
    for c in ["cpu", "gpu", "dram", "ane"]:
        comp[c + "_frac"] = comp[f"train_{c}_joules"] / tot
    return comp


if __name__ == "__main__":
    meta = D.load_metadata(True)
    x = pd.read_csv(OUT / "xai_per_galaxy.csv")
    print(shortcut_baselines().round(3).to_string())
    t = full_xai_table(x); t.to_csv(OUT / "xai_full.csv", index=False); print(t.round(3).to_string())
    gc, area, sr, corr, nw, edges, ns = per_galaxy(x, meta)
    print(area.round(2)); print(sr.round(3)); print(corr.round(2)); print(nw); print(edges); print(ns)
    fig_per_galaxy(area, sr, ns)
    print(fig_data_check())
    print(energy_breakdown().round(3).to_string())
