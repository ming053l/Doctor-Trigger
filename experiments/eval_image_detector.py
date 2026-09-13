"""Canonical-trained image-only detector: full matrix on canonical test + fusion with the relation.

Usage: eval_image_detector.py [plain|aug]; aug reads checkpoints.image_detector_aug and writes image_detector_aug.csv."""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py, or export it (see config.example.yaml)"
    return v + rest
import sys, yaml, numpy as np, pandas as pd, torch
sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score
from src import models
MODE = sys.argv[1] if len(sys.argv) > 1 else "plain"   # plain: checkpoints.image_detector; aug: the deployed-recipe augmentation
assert MODE in ("plain", "aug"), MODE
CKPTS = _dt("DT_IMAGE_DETECTOR_AUG" if MODE == "aug" else "DT_IMAGE_DETECTOR")
OUT_CSV = _dt("DT_OUTPUTS", "/image_detector_aug.csv" if MODE == "aug" else "/image_detector.csv")
R = _dt("DT_CANON_ROOT")
dev = "cuda"
cases = pd.read_csv(f"{R}/common/cases.csv")
cases = cases[cases.split == "test"]
cfg = yaml.safe_load(open(_dt("DT_MODEL_SNAPSHOT", "/config_ffpp_std_dfd_resnet50.yaml")))
HEADS = {}
for tag in ["ffpp", "dfdc", "celebdf"]:
    m = models.build_model(cfg).to(dev)
    m.load_state_dict(torch.load(CKPTS + f"/{tag}/best_val_auc.pt", map_location=dev, weights_only=False))
    HEADS[tag] = m.eval()
tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])
@torch.no_grad()
def prob(m, pil):
    o = m(tf(pil).unsqueeze(0).to(dev)); lg = o["logits"] if isinstance(o, dict) else o
    return float(torch.softmax(lg, 1)[0, 1]) if lg.ndim > 1 and lg.shape[1] == 2 else float(torch.sigmoid(lg.reshape(-1))[0])
rows = []
for _, q in cases.iterrows():
    pil = Image.open(q.input_path).convert("RGB")
    r = {"query_id": q.query_id, "dataset": q.dataset, "y_fake": q.y_fake}
    for tag, m in HEADS.items(): r[f"image_detector_{tag}"] = prob(m, pil)
    rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(OUT_CSV, index=False)
print("canonical-trained image-only detector, test AUC:")
for ds, g in d.groupby("dataset"):
    line = f"  {ds:8s}"
    for tag in HEADS: line += f"  {tag}={roc_auc_score(g.y_fake, g[f'image_detector_{tag}']):.4f}"
    print(line)
f = pd.read_csv(f"{R}/features/calibrated_features.csv", usecols=["query_id", "H"])
m = d.merge(f, on="query_id")
print("fusion max(in-domain image-only detector, 1-H):")
IDM = {"FF++": "ffpp", "DFDC": "dfdc", "CelebDF": "celebdf"}
for ds, g in m.groupby("dataset"):
    fus = np.maximum(g[f"image_detector_{IDM[ds]}"], 1 - g.H)
    print(f"  {ds:8s} H={roc_auc_score(g.y_fake, -g.H):.4f} image_detector={roc_auc_score(g.y_fake, g[f'image_detector_{IDM[ds]}']):.4f} fusion={roc_auc_score(g.y_fake, fus):.4f}")
