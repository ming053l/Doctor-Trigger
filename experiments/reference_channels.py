"""Reference baselines under the declared benign channels (FF++ test).

Distorts the benign/swap queries with the same channel families as the main
robustness table, then scores each reference verifier: amplitude-weighted phase
H, SSIM against the stored release, and ArcFace embedding distance."""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset"
    return v + rest
import sys, numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity as ssim
sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
sys.path.insert(0, _dt("DT_PEERS_ROOT"))
from scripts.benign_channels import transform_image
EPS = 1e-8
R = _dt("DT_CANON_ROOT")

def phase_of(pil):
    rgb = np.asarray(pil, dtype=np.float64)
    lum = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
    return np.fft.fftshift(np.angle(np.fft.fft2(lum)))

def H_of(pil, tp):
    with np.load(tp) as rec:
        host = rec["template_phase"].astype(np.float64); sup = rec["support"].astype(bool)
        w = rec["host_amplitude"].astype(np.float64)
    ph = phase_of(pil)
    if ph.shape != host.shape:
        pil = pil.resize((host.shape[1], host.shape[0])); ph = phase_of(pil)
    r = np.cos((ph - host + np.pi) % (2*np.pi) - np.pi)
    ww = np.where(sup, w, 0.0); return float(np.sum(r*ww) / (np.sum(ww) + EPS))

from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
fa.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)
def embed(pil):
    faces = fa.get(np.asarray(pil)[:, :, ::-1])
    return faces[0].normed_embedding if faces else None

FAMS = {"JPEG": ["jpeg_q90", "jpeg_q70", "jpeg_q50"],
        "blur": ["blur_k7"],
        "codec": ["x264_crf30", "x264_crf35", "x264_crf40"],
        "pixel": ["scale_0.5"]}
cases = pd.read_csv(f"{R}/common/cases.csv")
cases = cases[(cases.split == "test") & (cases.dataset == "FF++")]
emb_cache, rows = {}, []
for _, q in cases.iterrows():
    quer = Image.open(q.input_path).convert("RGB")
    rel = Image.open(q.doctor_trigger_released_path).convert("RGB")
    if q.enrollment_id not in emb_cache:
        emb_cache[q.enrollment_id] = embed(Image.open(q.enrollment_input_path).convert("RGB"))
    er = emb_cache[q.enrollment_id]
    for fam, conds in FAMS.items():
        for cond in conds:
            try:
                tq = transform_image(quer, cond, seed=(hash(cond) & 0xFFFF) + 17)
            except Exception:
                continue
            r = {"query_id": q.query_id, "fam": fam, "cond": cond, "y_fake": q.y_fake}
            r["H"] = H_of(tq, q.doctor_trigger_template_path)
            qa = np.asarray(tq.resize(rel.size).convert("L")); ra = np.asarray(rel.convert("L"))
            r["ssim"] = ssim(qa, ra)
            eq = embed(tq)
            r["arc"] = float(np.dot(er, eq)) if er is not None and eq is not None else np.nan
            rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(_dt("DT_OUTPUTS", "/reference_channels.csv"), index=False)
print(f"{'family':7s} {'H':>7s} {'SSIM':>7s} {'ArcFace':>8s} {'arc-nan':>7s}   (FF++, AUC over family)")
for fam, g in d.groupby("fam"):
    ga = g.dropna(subset=["arc"])
    print(f"{fam:7s} {roc_auc_score(g.y_fake, -g.H):7.4f} {roc_auc_score(g.y_fake, -g.ssim):7.4f} "
          f"{roc_auc_score(ga.y_fake, -ga.arc):8.4f} {g.arc.isna().sum():7d}")
