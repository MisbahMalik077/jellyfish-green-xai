"""Explanation quality against physical ground truth from the simulation.

For each true jellyfish in the test set we know, from the particles, where the stellar body is and
where extra-planar gas is (the tail region). Metrics:

  TEF   tail energy fraction       share of CAM mass outside the stellar body on gas pixels
  TEL   tail energy lift           TEF / (tail area fraction)   (1 = no better than uniform)
  GEF   gas energy fraction        share of CAM mass on gas pixels at all (vs empty background)
  PG    pointing game (gas)        CAM maximum lies on a gas pixel
  DEL   deletion AUC               mean P(jellyfish) as top-CAM pixels are removed (lower = more faithful)
  AGR   agreement with ResNet-50   Spearman rank correlation of CAMs (upsampled to 128x128)
"""
from __future__ import annotations

import time

import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.stats import spearmanr

from data import physical_masks
from energy import PowerMeter
from models import cam_target_layers, vit_reshape
from train import DEVICE, _sync

METHODS = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "scorecam": ScoreCAM}


def make_cam(method, model, name):
    kw = dict(model=model, target_layers=cam_target_layers(model, name))
    if name.startswith("vit"):
        kw["reshape_transform"] = vit_reshape
    cam = METHODS[method](**kw)
    if method == "scorecam":
        cam.batch_size = 16
    return cam


def compute_cams(model, name, method, x: torch.Tensor, bs=None):
    model.eval()
    bs = bs or (4 if method == 'scorecam' else 32)  # Score-CAM upsamples every channel: keep memory low
    cam = make_cam(method, model, name)
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(DEVICE)
        out.append(cam(input_tensor=xb, targets=[ClassifierOutputTarget(1)] * len(xb)))
    _sync()
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def deletion_auc(model, x: torch.Tensor, cams: np.ndarray, steps=(0, .02, .05, .1, .2, .3, .5)):
    """Remove (set to 0 = empty sky) the top-k% CAM pixels; area under P(jellyfish) curve."""
    model.eval()
    n, _, H, W = x.shape
    order = np.argsort(-cams.reshape(n, -1), axis=1)
    curves = []
    for f in steps:
        xm = x.clone().numpy().reshape(n, -1)
        k = int(f * H * W)
        if k:
            np.put_along_axis(xm, order[:, :k], 0.0, axis=1)
        xb = torch.from_numpy(xm.reshape(n, 1, H, W)).to(DEVICE)
        curves.append(torch.softmax(model(xb), 1)[:, 1].cpu().numpy())
    curves = np.stack(curves, 1)
    return np.trapz(curves, steps, axis=1) / (steps[-1] - steps[0]), curves


def physical_metrics(cams: np.ndarray, rows):
    out = []
    for c, r in zip(cams, rows):
        body, gas, tail = physical_masks(r)
        s = c.sum() + 1e-12
        tef = (c * tail).sum() / s
        area = tail.mean()
        iy, ix = np.unravel_index(np.argmax(c), c.shape)
        out.append(dict(TEF=tef, TEL=tef / area if area > 0 else np.nan, GEF=(c * gas).sum() / s,
                        PG=float(gas[iy, ix]), tail_area=area))
    return out


def explain_cost(model, name, method, x: torch.Tensor):
    """Energy/time per 1000 explanations."""
    compute_cams(model, name, method, x[:8])  # warm-up
    with PowerMeter() as pm:
        compute_cams(model, name, method, x)
    r = pm.result("xai_")
    return {"xai_per1k_seconds": r["xai_seconds"] * 1000 / len(x),
            "xai_per1k_joules": r["xai_joules"] * 1000 / len(x),
            "xai_per1k_dynamic_joules": r["xai_dynamic_joules"] * 1000 / len(x),
            "xai_energy_mode": r["xai_energy_mode"]}


def agreement(cam_a: np.ndarray, cam_b: np.ndarray):
    return np.array([spearmanr(a.ravel(), b.ravel()).statistic for a, b in zip(cam_a, cam_b)])
