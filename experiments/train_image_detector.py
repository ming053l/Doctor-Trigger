import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py, or export it (see config.example.yaml)"
    return v + rest
"""Retrain the passive head on the canonical pipeline's own data, per dataset.

Same architecture as the deployed image-only detector (build_model from the repo config); binary
benign-vs-swap on the canonical train queries, early stop on val AUC (DFDC uses a
10% train carve as val). Usage: train_image_detector.py <ffpp|dfdc|celebdf>
"""
import sys, yaml, numpy as np, pandas as pd, torch, os
sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from src import models

DS = sys.argv[1]
DSU = {"ffpp": "FF++", "dfdc": "DFDC", "celebdf": "CelebDF"}[DS]
R = _dt("DT_CANON_ROOT")
OUT = _dt("DT_IMAGE_DETECTOR") + f"/{DS}"
os.makedirs(OUT, exist_ok=True)
dev = "cuda"

cases = pd.read_csv(f"{R}/common/cases.csv")
cases = cases[(cases.dataset == DSU) & (cases.protocol.isin(["benign", "swap"]))]
tr = cases[cases.split == "train"]
va = cases[cases.split == "val"]
if len(va) == 0:
    groups = tr.enrollment_id.unique()
    cut = groups[-max(len(groups)//10, 16):]
    va, tr = tr[tr.enrollment_id.isin(cut)], tr[~tr.enrollment_id.isin(cut)]

aug = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(),
                 T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor()])
ev = T.Compose([T.Resize((224, 224)), T.ToTensor()])

class Q(Dataset):
    def __init__(self, df, tf): self.r = df.reset_index(drop=True); self.tf = tf
    def __len__(self): return len(self.r)
    def __getitem__(self, i):
        row = self.r.iloc[i]
        return self.tf(Image.open(row.input_path).convert("RGB")), int(row.y_fake)

cfg = yaml.safe_load(open(_dt("DT_MODEL_SNAPSHOT", "/config_ffpp_std_dfd_resnet50.yaml")))
m = models.build_model(cfg).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
lossf = torch.nn.CrossEntropyLoss()
tl = DataLoader(Q(tr, aug), batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
vl = DataLoader(Q(va, ev), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

def probs(loader):
    m.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            o = m(x.to(dev)); lg = o["logits"] if isinstance(o, dict) else o
            p = torch.softmax(lg, 1)[:, 1] if lg.ndim > 1 and lg.shape[1] == 2 else torch.sigmoid(lg.reshape(-1))
            ps += p.cpu().tolist(); ys += y.tolist()
    return ys, ps

best = 0
for ep in range(1, 21):
    m.train()
    for x, y in tl:
        opt.zero_grad()
        o = m(x.to(dev)); lg = o["logits"] if isinstance(o, dict) else o
        lossf(lg, y.to(dev)).backward(); opt.step()
    ys, ps = probs(vl); auc = roc_auc_score(ys, ps)
    print(f"epoch {ep}: val AUC {auc:.4f}", flush=True)
    if auc > best:
        best = auc; torch.save(m.state_dict(), f"{OUT}/best_val_auc.pt")
print(f"DONE {DS} best val AUC {best:.4f}")
