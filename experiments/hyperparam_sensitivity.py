"""Hyperparameter sensitivity beyond release strength (App. sensitivity, Fig. hyper):
(a) threshold FRR target: tau_rel from train-benign quantile at target t, test TPR and
    realized FPR on FF++ (closed-form relation H, the deployed reading);
(b) calibration-set size: realized test FPR spread when tau is set from n train-benign
    samples (200 resamples per n), target 1%;
(c) record-model loss weight lambda_key: retrain the full record model (3 seeds) per value, report
    S_rel AUC (y_rel) and S_key AUC (y_keycorr) on FF++ test and Celeb-DF test.
Writes hyper_sensitivity.json next to $DT_OUT.
"""
import os as _dtos, sys, json
def _dt(env, rest=""):
    v = _dtos.environ.get(env); assert v, f"{env} unset"; return v + rest
import numpy as np, pandas as pd
from pathlib import Path

R = _dt("DT_CANON_ROOT")
OUT = Path(_dtos.environ.get("DT_OUT", "."))
F = pd.read_csv(_dtos.environ.get("DT_FEATURES_CSV", f"{R}/features/calibrated_features.csv"))
out = {}

# ---------- (a)+(b): closed-form relation thresholding ----------
ff = F[F.dataset == "FF++"]
tr_ben = ff[(ff.split == "train") & (ff.protocol == "benign")].H.to_numpy()
te = ff[ff.split == "test"]
te_ben = te[te.protocol == "benign"].H.to_numpy()
te_fake = te[te.protocol == "swap"].H.to_numpy()
targets = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2]
out["frr_target"] = []
for t in targets:
    tau = float(np.quantile(tr_ben, t))
    out["frr_target"].append(dict(target=t, tau=tau,
        tpr=float((te_fake < tau).mean()), realized_fpr=float((te_ben < tau).mean())))
rng = np.random.default_rng(0)
out["cal_size"] = []
for n in [25, 50, 100, 200, 400]:
    fprs = []
    for _ in range(200):
        sub = rng.choice(tr_ben, size=n, replace=False)
        tau = np.quantile(sub, 0.01)
        fprs.append(float((te_ben < tau).mean()))
    q = np.quantile(fprs, [0.1, 0.25, 0.5, 0.75, 0.9])
    out["cal_size"].append(dict(n=n, q10=q[0], q25=q[1], med=q[2], q75=q[3], q90=q[4]))
print("closed-form sweeps done", flush=True)

# ---------- (c): lambda_key retraining ----------
sys.path.insert(0, _dt("DT_REPO_ROOT", "/workflow/record_model"))
import torch, random
from src.record_model import RecordModelConfig, RecordModelMLP, feature_columns_for_variant, multitask_loss, safe_auc

def standardize(x, mean=None, std=None):
    if mean is None:
        mean = np.nanmean(x, axis=0); std = np.nanstd(x, axis=0); std[std < 1e-8] = 1.0
    return ((np.nan_to_num(x, nan=mean) - mean) / std).astype(np.float32), mean, std

cols = feature_columns_for_variant("full_record_model", list(F.columns))
tr = F[(F.dataset == "FF++") & (F.split == "train")].reset_index(drop=True)
evals = {"FF++": F[(F.dataset == "FF++") & (F.split == "test")].reset_index(drop=True),
         "CelebDF": F[(F.dataset == "CelebDF") & (F.split == "test")].reset_index(drop=True)}
xtr_raw = tr[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
xtr, mu, sd = standardize(xtr_raw)
ttr = {k: torch.tensor(tr[v].to_numpy(np.float32)) for k, v in
       [("y_fake", "y_fake"), ("y_rel", "y_rel"), ("y_keycorr", "y_keycorr")]}
ftr = {n: torch.tensor(pd.to_numeric(tr[n], errors="coerce").fillna(0.0).to_numpy(np.float32))
       for n in ["G_s", "G_wrong_max"] if n in tr.columns}
xt = torch.tensor(xtr)
out["lambda_key"] = []
for lam in [0.0, 0.1, 0.3, 1.0, 3.0]:
    for seed in [0, 1, 2]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        cfg = RecordModelConfig(variant="full_record_model", input_dim=xtr.shape[1], hidden_dim=64,
                         lambda_key=lam)
        model = RecordModelMLP(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        model.train()
        for _ in range(1000):
            opt.zero_grad(set_to_none=True)
            loss = multitask_loss(model(xt), ttr, cfg, ftr)
            loss.backward(); opt.step()
        model.eval()
        for name, ev in evals.items():
            xe, _, _ = standardize(ev[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64), mu, sd)
            with torch.no_grad():
                o = {k: torch.sigmoid(v).numpy() for k, v in model(torch.tensor(xe)).items()}
            out["lambda_key"].append(dict(lam=lam, seed=seed, eval=name,
                rel_auc=safe_auc(ev.y_rel.to_numpy(int), o["S_rel"]),
                key_auc=safe_auc(ev.y_keycorr.to_numpy(int), o["S_key"])))
        print(f"lam={lam} seed={seed} done", flush=True)

(OUT / "hyper_sensitivity.json").write_text(json.dumps(out, indent=1))
print("ALL_DONE")
