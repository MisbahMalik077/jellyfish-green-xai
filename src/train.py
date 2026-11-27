"""Training + evaluation + energy accounting for one (model, seed) run."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from energy import PowerMeter
from models import build, count_flops, count_params

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
CKPT = ROOT / "checkpoints"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")


def _sync():
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    elif DEVICE.type == "cuda":
        torch.cuda.synchronize()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def predict(model, ds, bs=128):
    model.eval()
    probs = []
    for x, _ in DataLoader(ds, batch_size=bs):
        probs.append(torch.softmax(model(x.to(DEVICE)), 1)[:, 1].cpu())
    _sync()
    return torch.cat(probs).numpy()


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    conf = np.where(p >= 0.5, p, 1 - p)
    pred = (p >= 0.5).astype(int)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y, yhat), f1=f1_score(y, yhat), precision=precision_score(y, yhat),
                recall=recall_score(y, yhat), roc_auc=roc_auc_score(y, p), ece=ece(y, p))


def train_one(name, seed, train_ds, val_ds, test_ds, max_epochs=40, patience=6, bs=64, overwrite=False):
    RUNS.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{name}_s{seed}.json"
    if out.exists() and not overwrite:
        return json.loads(out.read_text())

    set_seed(seed)
    model = build(name).to(DEVICE)
    lr = 1e-3 if name == "small_cnn" else 3e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=g, drop_last=True)
    vl = DataLoader(val_ds, batch_size=128)

    best, best_state, bad, hist = np.inf, None, 0, []
    with PowerMeter() as pm_train:
        for ep in range(max_epochs):
            model.train()
            for x, y in tl:
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(x.to(DEVICE)), y.to(DEVICE))
                loss.backward()
                opt.step()
            sched.step()
            model.eval()
            vloss, n = 0.0, 0
            with torch.no_grad():
                for x, y in vl:
                    vloss += lossf(model(x.to(DEVICE)), y.to(DEVICE)).item() * len(y)
                    n += len(y)
            vloss /= n
            hist.append(vloss)
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        _sync()
    model.load_state_dict(best_state)
    torch.save(best_state, CKPT / f"{name}_s{seed}.pt")

    # inference energy: repeat test-set inference for >= 20 s for a stable power reading
    predict(model, test_ds)  # warm-up
    n_img, t0 = 0, time.time()
    with PowerMeter() as pm_inf:
        while time.time() - t0 < 20:
            predict(model, test_ds)
            n_img += len(test_ds)
    inf = pm_inf.result("inf_")
    per1k = {k.replace("inf_", "inf_per1k_"): v * 1000 / n_img for k, v in inf.items()
             if k in ("inf_seconds", "inf_joules", "inf_dynamic_joules", "inf_kwh", "inf_kgco2e")}

    p = predict(model, test_ds)
    y = test_ds.y
    res = dict(model=name, seed=seed, epochs=len(hist), best_val_loss=best, params=count_params(model),
               gflops=count_flops(build(name, pretrained=False)) / 1e9,
               **metrics(y, p), **pm_train.result("train_"), **per1k, inf_energy_mode=pm_inf.mode,
               test_keys=list(test_ds.df.key), test_probs=p.tolist())
    out.write_text(json.dumps(res))
    return res
