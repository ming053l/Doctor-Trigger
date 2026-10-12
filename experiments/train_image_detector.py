"""Retrain the passive head on the canonical pipeline's own data, per dataset.

Same architecture as the deployed image-only detector (experiments/shared/image_only_detector.py,
ImageNet-initialized backbone); binary benign-vs-swap on the canonical train queries, early stop
on val AUC (DFDC uses a 10% train carve as val). This plain variant augments with flips and
colour jitter only; train_image_detector_aug.py adds the deployed distortion recipe.
Writes <checkpoints.image_detector>/<ds>/best_val_auc.pt.
Usage: train_image_detector.py <ffpp|dfdc|celebdf>
"""
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
from doctor_trigger import config  # noqa: E402
from doctor_trigger.manifests import DATASET_TAGS, read_cases  # noqa: E402
from doctor_trigger.metrics import roc_auc  # noqa: E402
from experiments.shared.image_only_detector import INPUT_SIZE, build_image_only_detector  # noqa: E402

DEVICE = "cuda"
EPOCHS = 20
PLAIN_AUGMENTATION = T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), T.RandomHorizontalFlip(),
                                T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor()])
EVAL_TRANSFORM = T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor()])


class QueryImages(Dataset):
    """(transformed query image, y_fake) for each row of a cases.csv frame."""

    def __init__(self, df, tf): self.r = df.reset_index(drop=True); self.tf = tf
    def __len__(self): return len(self.r)

    def __getitem__(self, i):
        row = self.r.iloc[i]
        return self.tf(Image.open(row.input_path).convert("RGB")), int(row.y_fake)


def train_val_split(dataset_label):
    """Train / val queries (benign and swap) of one dataset; without a val split, the last
    10% of train enrollments (at least 16) become val."""
    cases = read_cases(dataset=dataset_label, protocol=["benign", "swap"])
    tr = cases[cases.split == "train"]
    va = cases[cases.split == "val"]
    if len(va) == 0:
        groups = tr.enrollment_id.unique()
        cut = groups[-max(len(groups)//10, 16):]
        va, tr = tr[tr.enrollment_id.isin(cut)], tr[~tr.enrollment_id.isin(cut)]
    return tr, va


def logits_of(output):
    return output["logits"] if isinstance(output, dict) else output


def probs(m, loader):
    m.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            lg = logits_of(m(x.to(DEVICE)))
            p = torch.softmax(lg, 1)[:, 1] if lg.ndim > 1 and lg.shape[1] == 2 else torch.sigmoid(lg.reshape(-1))
            ps += p.cpu().tolist(); ys += y.tolist()
    return ys, ps


def train(tag, augmentation, checkpoint_root):
    """Train on dataset ``tag`` (ffpp | dfdc | celebdf) and keep the best-val-AUC weights."""
    dataset_label = {v: k for k, v in DATASET_TAGS.items()}[tag]
    out = f"{checkpoint_root}/{tag}"
    os.makedirs(out, exist_ok=True)
    tr, va = train_val_split(dataset_label)
    m = build_image_only_detector(pretrained_backbone=True).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    tl = DataLoader(QueryImages(tr, augmentation), batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    vl = DataLoader(QueryImages(va, EVAL_TRANSFORM), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    best = 0
    for ep in range(1, EPOCHS + 1):
        m.train()
        for x, y in tl:
            opt.zero_grad()
            lossf(logits_of(m(x.to(DEVICE))), y.to(DEVICE)).backward(); opt.step()
        ys, ps = probs(m, vl); auc = roc_auc(ys, ps)
        print(f"epoch {ep}: val AUC {auc:.4f}", flush=True)
        if auc > best:
            best = auc; torch.save(m.state_dict(), f"{out}/best_val_auc.pt")
    print(f"DONE {tag} best val AUC {best:.4f}")


if __name__ == "__main__":
    train(sys.argv[1], PLAIN_AUGMENTATION, config.path("checkpoints.image_detector"))
