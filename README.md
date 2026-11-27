# Same Answer, Same Reason? — Green & explainable jellyfish galaxy classification (TNG50)

Paper target: **INSPIRE-STEMAL 2027** (Springer, special session *Intelligence 6.0: Sustainable Computational Systems* / Track 7). Deadline **15 Nov 2026**, 12 pages LNCS, double-blind.

## Layout
```
data/raw/jellyfish_TNG50-1.hdf5   Cosmological Jellyfish labels (Zinger+2024), from the TNG API
data/processed/maps/*.npz         128x128 gas + stellar maps we render from particle cutouts
data/processed/metadata.jsonl     per-galaxy catalogue values (mass, SFR, r_half, ...)
src/tng_data.py                   sample selection + serial API download + rendering (resumable)
src/data.py                       dataset, galaxy-level splits, physical body/tail masks
src/models.py                     6 models, FLOPs, CAM target layers
src/energy.py                     energy measurement (Zeus on-chip counters; CodeCarbon fallback)
src/train.py, src/xai.py          training/eval/energy, explanation metrics
notebooks/01..05                  run in order -> results/ (csv, tables, figures)
paper/                            LNCS LaTeX (compile in Overleaf: upload the folder + results/figures)
```

## Run
Open Jupyter and select kernel **"Python (jellyfish_green_xai)"**:
```
cd ~/jellyfish_green_xai && jupyter lab
```
Run notebooks 01 → 05 in order (05 = shortcut baselines, faithfulness/stability, per-galaxy analysis; no training). The API key is read from `.env` (`TNG_API_KEY=...`). Never commit or share that file.

## Energy measurement
Energy is read from the M3 Pro's on-chip counters (CPU, GPU, DRAM, ANE) with Zeus (`zeus-apple-silicon`), with
no admin rights needed. (CodeCarbon reports 0 W for the Apple GPU without root, so it is only a fallback.)
During measurement keep the laptop on mains power, with other apps closed, and avoid using it.
