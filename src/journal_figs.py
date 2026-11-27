"""All figures, LaTeX tables and headline numbers for the journal version.

Reads results/ (main runs) and results/journal/ (journal experiments). Every block is skipped if its
inputs are not there yet, so the script can be re-run as experiments finish.
Outputs: results/journal/figures/*.pdf|png, results/journal/tables/*.tex, results/journal/numbers.json
"""
from __future__ import annotations

import json
import shutil
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import data as D
from models import MODELS, PRETTY

warnings.filterwarnings("ignore")
R = D.ROOT / "results"; J = R / "journal"
FIG = J / "figures"; TAB = J / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
NUM: dict = {}
CI = 0.20705  # kg CO2e / kWh, UK DESNZ 2024 (electricity generated)

# reference categorical palette (fixed order, validated): one colour per model, everywhere
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
COL = dict(zip(MODELS, PAL))
MARK = dict(zip(MODELS, ["o", "s", "D", "^", "v", "P"]))
METH_PRETTY = {"gradcam": "Grad-CAM", "gradcam++": "Grad-CAM++", "scorecam": "Score-CAM", "gradcam_mid": "Grad-CAM (8×8)",
               "ig": "Integrated Gradients", "occlusion": "Occlusion"}
METH_COL = {"gradcam": "#2a78d6", "gradcam++": "#eb6834", "scorecam": "#1baf7a", "gradcam_mid": "#eda100",
            "ig": "#e87ba4", "occlusion": "#6250d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dedcd6"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6, "savefig.dpi": 200})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", metadata={"Creator": None, "Producer": None})
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def have(*paths):
    return all(p.exists() for p in paths)


def boot(v, n=2000, seed=0):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def model_legend(ax, models=MODELS, **kw):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[m], marker=MARK[m], ls="", ms=6, label=PRETTY[m]) for m in models]
    ax.legend(handles=h, **kw)


# ============================================================================ data description
def dataset_tables():
    meta = D.load_metadata(True)
    sp = pd.read_csv(D.ROOT / "data/processed/splits.csv")[["key", "split"]]
    meta = meta.merge(sp, on="key")
    # redshift from snapshot (TNG output list)
    zsnap = {67: 0.50, 72: 0.40, 78: 0.30, 84: 0.20, 91: 0.10, 99: 0.0}
    meta["log_ms"] = np.log10(meta.mass_stars * 1e10 / 0.6774)
    meta["log_fg"] = np.log10(meta.mass_gas / meta.mass_stars)
    meta["kpc_px"] = 2 * meta.half_width / 128 / 0.6774  # comoving kpc per pixel (h = 0.6774)
    rows = []
    for lab, name in ((1, "Jellyfish"), (0, "Non-jellyfish")):
        d = meta[meta.label == lab]
        q = lambda s: f"{s.median():.2f} [{s.quantile(.16):.2f}, {s.quantile(.84):.2f}]"
        rows.append((name, len(d), d.branch.nunique(), (d.split == "train").sum(), (d.split == "val").sum(),
                     (d.split == "test").sum(), q(d.log_ms), q(d.log_fg), q(d.score), f"{d.snap.min()}--{d.snap.max()}"))
    lines = [r"\begin{tabular}{lrrrrrccc}", r"\hline",
             r"Class & Images & Galaxies & Train & Val & Test & $\log M_\star/{\rm M_\odot}$ & $\log M_{\rm gas}/M_\star$ & Volunteer score \\",
             r"\hline"]
    for r in rows:
        lines.append(" & ".join(str(v) for v in r[:9]) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    (TAB / "tab_dataset.tex").write_text("\n".join(lines))
    NUM["dataset"] = dict(n=len(meta), n_gal=int(meta.branch.nunique()), kpc_px_med=float(meta.kpc_px.median()),
                          kpc_px_range=[float(meta.kpc_px.min()), float(meta.kpc_px.max())],
                          snap_range=[int(meta.snap.min()), int(meta.snap.max())],
                          logms_range=[float(meta.log_ms.quantile(.02)), float(meta.log_ms.quantile(.98))])
    for f in ("fig_examples", "fig_data_check", "fig_cams"):
        shutil.copy(R / "figures" / f"{f}.pdf", FIG / f"{f}.pdf")


# ============================================================================ accuracy & energy
def accuracy_energy():
    r = pd.read_csv(R / "runs.csv")
    g = r.groupby("model")
    t = pd.DataFrame({"acc": g.accuracy.mean(), "acc_sd": g.accuracy.std(ddof=1), "auc": g.roc_auc.mean(),
                      "auc_sd": g.roc_auc.std(ddof=1), "rec": g.recall.mean(), "ece": g.ece.mean(),
                      "params": g.params.first() / 1e6, "gflops": g.gflops.first(), "epochs": g.epochs.mean(),
                      "t": g.train_seconds.mean(), "E": g.train_joules.mean() / 1e3, "E_sd": g.train_joules.std(ddof=1) / 1e3,
                      "inf": g.inf_per1k_joules.mean(), "W": g.train_mean_watts.mean()}).loc[MODELS]
    t["co2_mg"] = t.E * 1e3 / 3.6e6 * CI * 1e6  # mg CO2e per training run
    lines = [r"\begin{tabular}{lrrccccrrrr}", r"\hline",
             r"Model & Params & GFLOPs & Accuracy & ROC-AUC & Recall & ECE & Epochs & Train (kJ) & CO$_2$e (mg) & Infer. (J/1k) \\", r"\hline"]
    for m, x in t.iterrows():
        lines.append(f"{PRETTY[m]} & {x.params:.2f}M & {x.gflops:.3f} & {x.acc:.3f}$\\pm${x.acc_sd:.3f} & {x.auc:.3f}$\\pm${x.auc_sd:.3f} & "
                     f"{x.rec:.3f} & {x.ece:.3f} & {x.epochs:.1f} & {x.E:.2f}$\\pm${x.E_sd:.2f} & {x.co2_mg:.0f} & {x.inf:.1f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (TAB / "tab_accuracy.tex").write_text("\n".join(lines))
    NUM["acc"] = t.round(4).to_dict(orient="index")
    NUM["total_train_kJ"] = float(r.train_joules.sum() / 1e3)
    NUM["total_train_gco2"] = float(r.train_joules.sum() / 3.6e6 * CI * 1e3)

    xa = pd.read_csv(R / "xai_per_galaxy.csv")
    gc = xa[xa.method == "gradcam"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for m in MODELS:
        x = t.loc[m]
        ax[0].errorbar(x.E, x.acc, yerr=x.acc_sd, xerr=x.E_sd, color=COL[m], marker=MARK[m], ms=7, lw=1.2, capsize=0)
        mu, lo, hi = boot(gc[gc.model == m].groupby("key").TEL.mean())
        ax[1].errorbar(x.E, mu, yerr=[[mu - lo], [hi - mu]], color=COL[m], marker=MARK[m], ms=7, lw=1.2)
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("Measured training energy (kJ)")
        a.set_xticks([0.3, 0.5, 1, 2, 3], ["0.3", "0.5", "1", "2", "3"]); a.minorticks_off()
    ax[0].set_ylabel("Test accuracy"); ax[1].set_ylabel("Grad-CAM tail lift (TEL)")
    ax[1].axhline(1, ls="--", lw=1, color=MUTED); ax[1].text(2.2, 1.1, "uniform map", fontsize=7, color=MUTED)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=COL[m], marker=MARK[m], ls="", ms=6, label=PRETTY[m]) for m in MODELS],
               ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    for a, l in zip(ax, "ab"):
        a.set_title(f"({l})", loc="left", fontsize=9)
    fig.tight_layout(); save(fig, "fig_pareto")

    # energy components
    comp = r.groupby("model")[["train_cpu_joules", "train_gpu_joules", "train_dram_joules", "train_ane_joules"]].mean().loc[MODELS] / 1e3
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    left = np.zeros(len(MODELS)); names = ["CPU", "GPU", "DRAM", "ANE"]; cc = ["#eb6834", "#2a78d6", "#1baf7a", "#eda100"]
    for c, n, col in zip(comp.columns, names, cc):
        ax.barh(range(len(MODELS)), comp[c], left=left, color=col, label=n, height=0.62, edgecolor="white", lw=1)
        left += comp[c].values
    ax.set_yticks(range(len(MODELS)), [PRETTY[m] for m in MODELS]); ax.invert_yaxis()
    ax.set_xlabel("Training energy per run (kJ)"); ax.legend(fontsize=7, ncol=4, loc="lower right", bbox_to_anchor=(1, 1.0))
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); save(fig, "fig_energy_components")
    NUM["gpu_frac"] = (comp.train_gpu_joules / comp.sum(1)).round(3).to_dict()
    NUM["dram_frac"] = (comp.train_dram_joules / comp.sum(1)).round(3).to_dict()


# ============================================================================ explanation methods (stage A, H)
def methods():
    fa = sorted(J.glob("A_*_s*.csv"))
    if not fa:
        return
    A = pd.concat([pd.read_csv(f) for f in fa])
    old = pd.read_csv(R / "xai_per_galaxy.csv")
    old = old[old.method.isin(["gradcam++", "scorecam"])]
    allm = pd.concat([A, old], ignore_index=True)
    order = ["gradcam", "gradcam++", "scorecam", "gradcam_mid", "ig", "occlusion"]
    agg = allm.groupby(["model", "method"]).agg(TEL=("TEL", "mean"), TEL_gas=("TEL_gas", "mean"), PG=("PG", "mean"),
                                                DEL=("DEL", "mean")).reset_index()
    NUM["methods"] = {f"{r.model}|{r.method}": dict(TEL=r.TEL, TEL_gas=r.TEL_gas, PG=r.PG, DEL=r.DEL) for r in agg.itertuples()}
    # bootstrap CI on gas-normalised lift (galaxy level, seeds averaged)
    ci = {}
    for (m, me), d in A.groupby(["model", "method"]):
        ci[(m, me)] = boot(d.groupby("key").TEL_gas.mean())
    NUM["TEL_gas_ci"] = {f"{k[0]}|{k[1]}": v for k, v in ci.items()}

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=False)
    xs = np.arange(len(MODELS))
    panels = [("TEL", ["gradcam", "gradcam++", "scorecam", "gradcam_mid"], "Tail lift (TEL) — CAM methods", True),
              ("TEL_gas", ["gradcam", "gradcam_mid", "ig", "occlusion"], "Gas-normalised tail lift", True),
              ("DEL", order, "Deletion AUC (lower = more faithful)", False)]
    for a, (col, meths, lab, ref) in zip(ax, panels):
        k = len(meths)
        for j, me in enumerate(meths):
            d = agg[agg.method == me].set_index("model").reindex(MODELS)
            a.plot(xs + (j - (k - 1) / 2) * 0.14, d[col], ls="", marker="o", ms=5, color=METH_COL[me], label=METH_PRETTY[me])
        if ref:
            a.axhline(1, ls="--", lw=1, color=MUTED)
        a.set_xticks(xs, [PRETTY[m] for m in MODELS], rotation=40, ha="right", fontsize=7)
        a.set_title(lab, fontsize=8.5); a.grid(axis="x", visible=False)
    ax[0].set_yscale("log"); ax[0].set_yticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
    ax[1].set_ylim(0.6, 1.4)
    handles = {}
    for a in ax:
        for h, l in zip(*a.get_legend_handles_labels()):
            handles[l] = h
    fig.legend(handles.values(), handles.keys(), ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(); save(fig, "fig_methods")

    # table of new methods
    lines = [r"\begin{tabular}{lcccccccccccc}", r"\hline",
             r" & \multicolumn{4}{c}{Gas-normalised tail lift $\uparrow$} & \multicolumn{4}{c}{Gas pointing (PG) $\uparrow$} & \multicolumn{4}{c}{Deletion AUC $\downarrow$} \\",
             r"Model & GC & GC$_{8}$ & IG & Occ. & GC & GC$_{8}$ & IG & Occ. & GC & GC$_{8}$ & IG & Occ. \\", r"\hline"]
    for m in MODELS:
        cells = []
        for col in ("TEL_gas", "PG", "DEL"):
            for me in ("gradcam", "gradcam_mid", "ig", "occlusion"):
                v = agg[(agg.model == m) & (agg.method == me)][col]
                cells.append(f"{v.iloc[0]:.2f}" if len(v) else "--")
        lines.append(PRETTY[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    (TAB / "tab_methods.tex").write_text("\n".join(lines))

    # example maps (seed 0, first 12 test jellyfish)
    if have(J / "A_maps_seed0.npy"):
        maps = np.load(J / "A_maps_seed0.npy", allow_pickle=True).item()
        sp = pd.read_csv(D.ROOT / "data/processed/splits.csv"); te = sp[(sp.split == "test") & (sp.label == 1)].reset_index(drop=True)
        meta = D.load_metadata(True).set_index("key")
        def offset(k):  # distance of the gas centroid from the stellar centre, in units of the gas extent
            g = D.load_maps(k)[0]; yy, xx = np.mgrid[:128, :128] - 63.5
            w = g / g.sum(); cx, cy = (w * xx).sum(), (w * yy).sum()
            return np.hypot(cx, cy) / np.sqrt((w * (xx ** 2 + yy ** 2)).sum())
        gi = int(np.argmax([offset(k) for k in te.key[:12]]))  # clearest one-sided tail
        NUM["map_example_key"] = te.key[gi]
        key = te.key[gi]; gas = D.to_image(D.load_maps(key)[0]); body, gm, tail = D.physical_masks(_row(meta, key))
        show = ["resnet18", "mobilenetv3_small_100", "vit_tiny_patch16_224"]
        cols = ["gradcam", "gradcam_mid", "ig", "occlusion"]
        fig, ax = plt.subplots(len(show), len(cols) + 1, figsize=(7.2, 4.4))
        for i, m in enumerate(show):
            ax[i, 0].imshow(gas, cmap="magma", origin="lower")
            ax[i, 0].contour(body, [0.5], colors="white", linewidths=0.8, origin="lower")
            ax[i, 0].contour(tail, [0.5], colors="#5fd4ff", linewidths=0.6, origin="lower")
            ax[i, 0].set_ylabel(PRETTY[m], fontsize=8)
            for j, me in enumerate(cols):
                a = ax[i, j + 1]
                if (m, me) in maps:
                    c = maps[(m, me)][gi].astype(np.float32)
                    if me == "ig":
                        from scipy.ndimage import gaussian_filter
                        c = gaussian_filter(c, 2)  # display only
                    c = c / (c.max() + 1e-12)
                    a.imshow(gas, cmap="gray", origin="lower"); a.imshow(c, cmap="viridis", alpha=0.6, origin="lower")
                else:
                    a.text(0.5, 0.5, "not applicable\n(single token grid)", ha="center", va="center", fontsize=7, color=MUTED, transform=a.transAxes)
                    for s in a.spines.values():
                        s.set_visible(False)
                if i == 0:
                    a.set_title(METH_PRETTY[me], fontsize=8)
            if i == 0:
                ax[i, 0].set_title("gas; body / tail", fontsize=8)
        for a in ax.ravel():
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
        fig.tight_layout(); save(fig, "fig_method_maps")

    if have(J / "H_agreement_occlusion.csv"):
        H = pd.read_csv(J / "H_agreement_occlusion.csv")
        h = H.groupby(["model", "method"]).agg(rho_all=("rho_all", "mean"), rho_gas=("rho_gas", "mean")).reset_index()
        NUM["agree_occ"] = {f"{r.model}|{r.method}": dict(all=r.rho_all, gas=r.rho_gas) for r in h.itertuples()}
        fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
        for a, col, ttl in zip(ax, ("rho_all", "rho_gas"), ("whole image", "gas pixels only")):
            for j, me in enumerate(["gradcam", "gradcam++", "ig"]):
                d = h[h.method == me].set_index("model").reindex(MODELS)
                a.plot(xs + (j - 1) * 0.18, d[col], ls="", marker="o", ms=6, color=METH_COL[me], label=METH_PRETTY[me])
            a.axhline(0, lw=1, color=MUTED); a.set_title(f"Spearman ρ with occlusion ({ttl})", fontsize=8.5)
            a.set_xticks(xs, [PRETTY[m] for m in MODELS], rotation=40, ha="right", fontsize=7); a.grid(axis="x", visible=False)
        ax[0].legend(fontsize=7); fig.tight_layout(); save(fig, "fig_agreement")


def _row(meta, key):
    r = meta.loc[key].copy(); r["key"] = key
    return r


# ============================================================================ sanity checks (B)
def sanity():
    if not have(J / "B_sanity.csv"):
        return
    B = pd.read_csv(J / "B_sanity.csv")
    NUM["sanity"] = B.round(3).to_dict(orient="records")
    lines = [r"\begin{tabular}{lcccccc}", r"\hline",
             r" & \multicolumn{3}{c}{Grad-CAM} & \multicolumn{3}{c}{Integrated Gradients} \\",
             r"Model & $|\rho|$ head & $|\rho|$ full & TEL full & $|\rho|$ head & $|\rho|$ full & TEL full \\", r"\hline"]
    for m in MODELS:
        c = []
        for me in ("gradcam", "ig"):
            d = B[(B.model == m) & (B.method == me)].set_index("randomised")
            c += [f"{d.loc['head','spearman']:.2f}", f"{d.loc['full','spearman']:.2f}", f"{d.loc['full','TEL_rand']:.2f}"]
        lines.append(PRETTY[m] + " & " + " & ".join(c) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    (TAB / "tab_sanity.tex").write_text("\n".join(lines))


# ============================================================================ cost (G, F)
def cost():
    C = pd.read_csv(R / "xai_cost.csv")
    r = pd.read_csv(R / "runs.csv"); inf = r.groupby("model").inf_per1k_joules.mean(); tr = r.groupby("model").train_joules.mean()
    rows = {m: {"inference": inf[m], "train": tr[m]} for m in MODELS}
    for x in C.itertuples():
        rows[x.model][x.method] = x.xai_per1k_joules
    if have(J / "G_cost.csv"):
        for x in pd.read_csv(J / "G_cost.csv").itertuples():
            rows[x.model][x.method] = x.per1k_joules
    if have(J / "F_cpu_inference.csv"):
        for x in pd.read_csv(J / "F_cpu_inference.csv").itertuples():
            rows[x.model]["inference_cpu"] = x.per1k_joules
    T = pd.DataFrame(rows).T.loc[MODELS]
    NUM["cost"] = T.round(2).to_dict(orient="index")
    cols = [c for c in ["inference", "inference_cpu", "gradcam", "gradcam++", "gradcam_mid", "ig", "occlusion", "scorecam"] if c in T]
    names = {"inference": "Inference (GPU)", "inference_cpu": "Inference (CPU)", **METH_PRETTY}
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "r}", r"\hline",
             "Model & " + " & ".join(names[c] for c in cols) + r" & SC (1k) / train \\", r"\hline"]
    for m in MODELS:
        lines.append(PRETTY[m] + " & " + " & ".join("--" if pd.isna(T.loc[m, c]) else f"{T.loc[m, c]:,.1f}".replace(",", "{,}") for c in cols)
                     + f" & {T.loc[m, 'scorecam'] / T.loc[m, 'train']:.2f}$\\times$ \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (TAB / "tab_cost.tex").write_text("\n".join(lines))
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    k = len(cols); w = 0.8 / k
    for j, c in enumerate(cols):
        col = {"inference": "#52514e", "inference_cpu": "#a9a79f"}.get(c, METH_COL.get(c))
        ax.bar(np.arange(len(MODELS)) + (j - (k - 1) / 2) * w, T[c], width=w * 0.9, color=col, label=names[c])
    ax.set_yscale("log"); ax.set_ylabel("Energy per 1,000 images (J)")
    ax.set_xticks(range(len(MODELS)), [PRETTY[m] for m in MODELS]); ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout(); save(fig, "fig_cost")


# ============================================================================ learning curves (E)
def learning():
    files = sorted(J.glob("E_runs_f*/*.json"))
    if not files:
        return
    E = pd.DataFrame([{k: v for k, v in json.loads(f.read_text()).items() if k not in ("test_keys", "test_probs")} for f in files])
    full = pd.read_csv(R / "runs.csv"); xa = pd.read_csv(R / "xai_per_galaxy.csv")
    tel = xa[xa.method == "gradcam"].groupby(["model", "seed"]).TEL.mean().rename("TEL_gradcam").reset_index()
    full = full.merge(tel, on=["model", "seed"]); full["frac"] = 1.0
    full["n_train"] = len(pd.read_csv(D.ROOT / "data/processed/splits.csv").query("split=='train'"))
    E = pd.concat([E, full[full.model.isin(E.model.unique())][E.columns.intersection(full.columns)]], ignore_index=True)
    g = E.groupby(["model", "frac"]).agg(acc=("accuracy", "mean"), acc_sd=("accuracy", "std"), E=("train_joules", "mean"),
                                         TEL=("TEL_gradcam", "mean"), n=("n_train", "mean")).reset_index()
    NUM["learning"] = g.round(4).to_dict(orient="records")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for m in [mm for mm in MODELS if mm in set(g.model)]:
        d = g[g.model == m].sort_values("frac")
        ax[0].errorbar(d.n, d.acc, yerr=d.acc_sd, color=COL[m], marker=MARK[m], lw=1.6, ms=5, label=PRETTY[m])
        ax[1].plot(d.n, d.E / 1e3, color=COL[m], marker=MARK[m], lw=1.6, ms=5)
        ax[2].plot(d.n, d.TEL, color=COL[m], marker=MARK[m], lw=1.6, ms=5)
    nt = sorted(g.n.unique())
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("Training images")
        a.set_xticks(nt, [f"{v:.0f}" for v in nt]); a.minorticks_off()
    ax[0].set_ylabel("Test accuracy"); ax[1].set_ylabel("Training energy (kJ)"); ax[1].set_yscale("log")
    ax[2].set_ylabel("Grad-CAM tail lift"); ax[2].axhline(1, ls="--", lw=1, color=MUTED)
    h, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h, l_, ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    for a, l in zip(ax, "abc"):
        a.set_title(f"({l})", loc="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, "fig_learning")


# ============================================================================ hold-out (C)
def holdout():
    if not have(J / "C_holdout_preds.csv"):
        return
    P = pd.read_csv(J / "C_holdout_preds.csv")
    runs = {(json.loads(f.read_text())["model"], json.loads(f.read_text())["seed"]): json.loads(f.read_text()) for f in (R / "runs").glob("*.json")}
    sp = pd.read_csv(D.ROOT / "data/processed/splits.csv")
    test = sp[sp.split == "test"][["key", "label", "score"]]
    res = []
    for m in MODELS:
        for s in range(3):
            d = P[(P.model == m) & (P.seed == s) & (P.subset == "highz")]
            yhat = d.p_jf >= 0.5
            rr = runs[(m, s)]; tp = pd.DataFrame(dict(key=rr["test_keys"], p=rr["test_probs"])).merge(test, on="key")
            res.append(dict(model=m, seed=s, hz_acc=(yhat == (d.label == 1)).mean(), hz_rec=yhat[d.label == 1].mean(),
                            hz_spec=(~yhat[d.label == 0]).mean(), id_acc=((tp.p >= .5) == (tp.label == 1)).mean(),
                            id_rec=(tp.p[tp.label == 1] >= .5).mean(), id_spec=(tp.p[tp.label == 0] < .5).mean()))
    hz = pd.DataFrame(res).groupby("model").agg(["mean", "std"]).loc[MODELS]
    NUM["highz"] = {m: {k: float(hz.loc[m, (k, "mean")]) for k in ["hz_acc", "hz_rec", "hz_spec", "id_acc", "id_rec", "id_spec"]} for m in MODELS}
    # intermediate scores: combine test set (labelled) + intermediate hold-out, seeds averaged
    mid = P[P.subset == "intermediate"].groupby(["model", "key"]).agg(p=("p_jf", "mean"), score=("score", "first")).reset_index()
    tpp = []
    for m in MODELS:
        ps = np.mean([runs[(m, s)]["test_probs"] for s in range(3)], 0)
        tpp.append(pd.DataFrame(dict(model=m, key=runs[(m, 0)]["test_keys"], p=ps)).merge(test[["key", "score"]], on="key"))
    allp = pd.concat([mid, pd.concat(tpp)])
    bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    allp["bin"] = pd.cut(allp.score, bins, include_lowest=True)
    curve = allp.groupby(["model", "bin"], observed=True).p.mean().unstack()
    rho = {m: float(spearmanr(mid[mid.model == m].score, mid[mid.model == m].p).statistic) for m in MODELS}
    NUM["intermediate_rho"] = rho; NUM["intermediate_n"] = int(mid.key.nunique())
    NUM["intermediate_curve"] = {m: curve.loc[m].round(3).tolist() for m in MODELS}
    if have(J / "C_highz_gradcam.csv"):
        hg = pd.read_csv(J / "C_highz_gradcam.csv")
        NUM["highz_TEL"] = hg.groupby("model").TEL.mean().round(3).to_dict()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1, 1.25]))
    y = np.arange(len(MODELS))
    for i, m in enumerate(MODELS):
        for k, (a_, b_, off) in enumerate((("id_rec", "hz_rec", -0.15), ("id_spec", "hz_spec", 0.15))):
            a0, b0 = hz.loc[m, (a_, "mean")], hz.loc[m, (b_, "mean")]
            ax[0].plot([a0, b0], [i + off] * 2, color=GRID, lw=2, zorder=1)
            ax[0].plot(a0, i + off, "o" if k == 0 else "s", mfc="white", mec=COL[m], ms=6, zorder=2)
            ax[0].plot(b0, i + off, "o" if k == 0 else "s", color=COL[m], ms=6, zorder=3)
    ax[0].set_yticks(y, [PRETTY[m] for m in MODELS]); ax[0].invert_yaxis()
    ax[0].set_xlabel("Recall (circles) / specificity (squares)")
    ax[0].set_title("(a) z ≤ 0.5 test (open) → z ≈ 0.5-2 unseen (filled)", loc="left", fontsize=8)
    mids = [(b.left + b.right) / 2 for b in curve.columns]
    for m in MODELS:
        ax[1].plot(mids, curve.loc[m].values, marker=MARK[m], color=COL[m], ms=4.5, lw=1.6, label=PRETTY[m])
    ax[1].axvspan(0.2, 0.8, color="#f3f2ee", zorder=0); ax[1].text(0.5, 0.97, "ambiguous scores (unseen galaxies)", ha="center", va="top", fontsize=7, color=MUTED, transform=ax[1].get_xaxis_transform())
    ax[1].set_xlabel("Volunteer score"); ax[1].set_ylabel("Mean predicted P(jellyfish)")
    ax[1].set_title("(b) model confidence vs. human agreement", loc="left", fontsize=8)
    ax[1].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(); save(fig, "fig_holdout")
    lines = [r"\begin{tabular}{lcccccc}", r"\hline",
             r" & \multicolumn{3}{c}{Test set ($z\leq0.5$)} & \multicolumn{3}{c}{Unseen galaxies, $z\approx0.5$--2} \\",
             r"Model & Accuracy & Recall & Specificity & Accuracy & Recall & Specificity \\", r"\hline"]
    for m in MODELS:
        f = lambda k: f"{hz.loc[m, (k, 'mean')]:.3f}"
        lines.append(f"{PRETTY[m]} & {f('id_acc')} & {f('id_rec')} & {f('id_spec')} & {f('hz_acc')} & {f('hz_rec')} & {f('hz_spec')} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (TAB / "tab_holdout.tex").write_text("\n".join(lines))


# ============================================================================ physical drivers (D)
def physics():
    hp = D.ROOT / "data/processed/hosts.jsonl"
    if not hp.exists():
        return
    hosts = pd.DataFrame([json.loads(l) for l in open(hp)]).drop_duplicates(["snap", "grnr"])
    meta = D.load_metadata(True).merge(hosts, on=["snap", "grnr"], how="left")
    box = 35000.0
    d = np.stack([meta.pos_x - meta.gx, meta.pos_y - meta.gy, meta.pos_z - meta.gz], 1)
    d = (d + box / 2) % box - box / 2
    meta["r_r200"] = np.linalg.norm(d, axis=1) / meta.r200c
    meta["log_m200"] = np.log10(meta.m200c * 1e10 / 0.6774)
    xa = pd.read_csv(R / "xai_per_galaxy.csv"); gc = xa[xa.method == "gradcam"]
    pg = gc.groupby(["model", "key"]).agg(TEL=("TEL", "mean"), p=("p_jf", "mean"), area=("tail_area", "first")).reset_index()
    pg = pg.merge(meta[["key", "log_m200", "r_r200", "score", "mass_stars"]], on="key")
    pg["log_ms"] = np.log10(pg.mass_stars * 1e10 / 0.6774); pg["hit"] = pg.p >= 0.5
    cor = {}
    for m in MODELS:
        d0 = pg[pg.model == m]
        cor[m] = {v: float(spearmanr(d0[v], d0.TEL, nan_policy="omit").statistic) for v in ["log_m200", "r_r200", "area", "score", "log_ms"]}
    NUM["phys_spearman_TEL"] = cor
    NUM["host_mass_range"] = [float(pg.log_m200.quantile(q)) for q in (0.05, 0.5, 0.95)]
    NUM["r_r200_range"] = [float(pg.r_r200.quantile(q)) for q in (0.05, 0.5, 0.95)]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for a, v, lab in zip(ax, ["log_m200", "r_r200", "log_ms"], [r"Host $\log M_{200c}/{\rm M_\odot}$", r"$r/R_{200c}$", r"$\log M_\star/{\rm M_\odot}$"]):
        q = pg[pg.model == MODELS[0]][v].quantile([0, 1 / 3, 2 / 3, 1]).to_numpy(copy=True)
        q[0] -= 1e-9; q[-1] += 1e-9
        for m in MODELS:
            d0 = pg[pg.model == m].copy(); d0["b"] = pd.cut(d0[v], q)
            s = d0.groupby("b", observed=True).TEL.mean()
            a.plot(range(len(s)), s.values, marker=MARK[m], color=COL[m], ms=4.5, lw=1.6, label=PRETTY[m])
        a.set_xticks(range(3), [f"{q[i]:.1f}–{q[i + 1]:.1f}" for i in range(3)], fontsize=6.5)
        a.set_xlabel(lab); a.axhline(1, ls="--", lw=1, color=MUTED); a.grid(axis="x", visible=False)
    ax[0].set_ylabel("Grad-CAM tail lift")
    h, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h, l_, ncol=6, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, "fig_physics")


def per_galaxy_redraw():
    xa = pd.read_csv(R / "xai_per_galaxy.csv"); meta = D.load_metadata(True)
    gc = xa[xa.method == "gradcam"].merge(meta[["key", "score"]], on="key")
    gc["area_bin"] = pd.qcut(gc.tail_area, 3, labels=["small", "medium", "large"])
    gc["score_bin"] = pd.cut(gc.score, [0.79, 0.9, 0.97, 1.01], labels=["0.80-0.90", "0.90-0.97", ">0.97"])
    gc["hit"] = gc.p_jf >= .5
    area = gc.groupby(["model", "area_bin"], observed=True).TEL.mean().unstack()
    rec = gc.groupby(["model", "score_bin"], observed=True).hit.mean().unstack()
    hm = gc.groupby(["model", "hit"]).TEL.mean().unstack()
    NUM["TEL_hit_miss"] = hm.round(3).to_dict(orient="index"); NUM["TEL_by_area"] = area.round(3).to_dict(orient="index")
    NUM["recall_by_score"] = rec.round(3).to_dict(orient="index")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for m in MODELS:
        ax[0].plot(range(3), area.loc[m], marker=MARK[m], color=COL[m], ms=5, lw=1.6, label=PRETTY[m])
        ax[1].plot(range(3), rec.loc[m], marker=MARK[m], color=COL[m], ms=5, lw=1.6)
    ax[0].axhline(1, ls="--", lw=1, color=MUTED); ax[0].set_yscale("log"); ax[0].set_yticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
    ax[0].set_xticks(range(3), ["small tail", "medium tail", "large tail"]); ax[0].set_ylabel("Grad-CAM tail lift")
    ax[1].set_xticks(range(3), list(rec.columns)); ax[1].set_xlabel("Volunteer score"); ax[1].set_ylabel("Recall on test jellyfish")
    ax[0].legend(fontsize=6.5, ncol=2)
    for a, l in zip(ax, "ab"):
        a.set_title(f"({l})", loc="left", fontsize=9); a.grid(axis="x", visible=False)
    fig.tight_layout(); save(fig, "fig_per_galaxy")


def old_tables():
    x = pd.read_csv(R / "xai_full.csv"); NUM["xai_full"] = x.round(4).to_dict(orient="records")
    s = pd.read_csv(R / "shortcut_baselines.csv"); NUM["shortcut"] = s.round(4).to_dict(orient="records")
    xa = pd.read_csv(R / "xai_per_galaxy.csv")
    NUM["xai_pg_agr"] = xa.groupby(["model", "method"])[["PG", "AGR"]].mean().round(3).reset_index().to_dict(orient="records")


if __name__ == "__main__":
    for f in (dataset_tables, accuracy_energy, old_tables, per_galaxy_redraw, methods, sanity, cost, learning, holdout, physics):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc(); print("FAILED", f.__name__, e)
    (J / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
