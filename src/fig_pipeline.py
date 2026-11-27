"""Schematic of the full pipeline (data -> models -> energy -> explanations -> physical scoring)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "journal" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED = "#0b0b0b", "#52514e"
FILL = {"data": "#e8f1fb", "model": "#fdeee7", "energy": "#fdf3dc", "xai": "#e6f6ef", "eval": "#efedfb"}
EDGE = {"data": "#2a78d6", "model": "#eb6834", "energy": "#c98500", "xai": "#1baf7a", "eval": "#6250d6"}


def box(ax, x, y, w, h, title, body, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.4))
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=8.0, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.068, body, ha="center", va="top", fontsize=6.6, color=MUTED, linespacing=1.35)


def arrow(ax, a, b, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=1.1, color=MUTED,
                                 connectionstyle=f"arc3,rad={rad}"))


fig, ax = plt.subplots(figsize=(8.4, 4.3))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
W, H = 0.215, 0.27
# row 1: data
box(ax, 0.01, 0.70, W, H, "TNG50-1 + labels",
    "Cosmological Jellyfish\ncitizen-science scores\n(Zinger et al. 2024)\nsnapshots 67-99, z ≤ 0.5", "data")
box(ax, 0.265, 0.70, W, H, "Leakage control",
    "branch-grouped splits\nnegatives matched in\nM★, Mgas/M★, snapshot\n992 + 992 images", "data")
box(ax, 0.52, 0.70, W, H, "Rendering",
    "particle cutouts (API)\nlog Σgas, 128×128 px\nFoV 15 r½,★ (20-80 ckpc/h)\nflip/rot90 augmentation", "data")
box(ax, 0.775, 0.70, W, H, "Physical masks",
    "stellar body: R < 2 r½,★\ngas: log Σgas > -5.5\ntail = gas ∖ body\n(independent of models)", "data")
# row 2: models + energy
box(ax, 0.01, 0.37, 0.30, H, "Six classifiers × 3 seeds",
    "ResNet-50 · ResNet-18 · EfficientNet-B0\nMobileNetV3-S · ViT-Tiny  (ImageNet)\nSmallCNN (from scratch)\nAdamW, cosine LR, early stopping", "model")
box(ax, 0.355, 0.37, 0.29, H, "On-chip energy (Zeus)",
    "Apple M3 Pro counters:\nCPU + GPU + DRAM + ANE\ntraining · inference · explanation\nJ → kWh → g CO₂e (DESNZ 2024)", "energy")
box(ax, 0.69, 0.37, 0.30, H, "Attribution methods",
    "Grad-CAM · Grad-CAM++ · Score-CAM\nGrad-CAM at 8×8 (mid layer)\nIntegrated Gradients · Occlusion\n+ randomisation sanity checks", "xai")
# row 3: evaluation
box(ax, 0.01, 0.03, 0.30, H, "Accuracy & calibration",
    "accuracy · ROC-AUC · recall · ECE\nscalar shortcut baselines\nlearning curves (10-100 %)", "eval")
box(ax, 0.355, 0.03, 0.29, H, "Explanation metrics",
    "tail lift (TEL), gas-normalised TEL\ngas pointing · deletion AUC\nagreement with occlusion\nseed stability", "eval")
box(ax, 0.69, 0.03, 0.30, H, "Generalisation & physics",
    "unseen galaxies at z ≈ 0.5-2\nambiguous scores (0.2-0.8)\nhost mass, r/R200, tail size\nCPU vs GPU energy", "eval")
for x0 in (0.225, 0.48, 0.735):
    arrow(ax, (x0, 0.835), (x0 + 0.04, 0.835))
arrow(ax, (0.62, 0.70), (0.16, 0.64), rad=0.0)
arrow(ax, (0.885, 0.70), (0.84, 0.64))
arrow(ax, (0.31, 0.505), (0.355, 0.505)); arrow(ax, (0.645, 0.505), (0.69, 0.505))
arrow(ax, (0.16, 0.37), (0.16, 0.30)); arrow(ax, (0.84, 0.37), (0.50, 0.30)); arrow(ax, (0.84, 0.37), (0.84, 0.30))
fig.savefig(OUT / "fig_pipeline.pdf", bbox_inches="tight", metadata={"Creator": None, "Producer": None})
fig.savefig(OUT / "fig_pipeline.png", dpi=200, bbox_inches="tight")
