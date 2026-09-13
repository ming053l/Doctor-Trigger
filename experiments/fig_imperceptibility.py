import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py, or export it (see config.example.yaml)"
    return v + rest
"""Qualitative imperceptibility figure: clean | watermarked (per method) | 10x residual.
Uses a real DT-enrolled trigger pair as the base face; peers embed on the same clean source."""
import sys, glob, json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"]="Times New Roman"
matplotlib.rcParams["mathtext.fontset"]="stix"
import matplotlib.pyplot as plt
from PIL import Image
import torch, torchvision.transforms as T

dev="cuda:0"; OUT=_dt("DT_OUTPUTS", "/figures")
os.makedirs(OUT, exist_ok=True)

# --- pick a DT trigger pair (clean source + DT-watermarked) ---
js=sorted(glob.glob(_dt("DT_LEGACY_TRIGGER_GLOB"),recursive=True))
pick=None
for j in js:
    d=json.load(open(j))
    if os.path.exists(d['source']) and os.path.exists(d['trigger_path']):
        pick=d; break
clean=Image.open(pick['source']).convert("RGB").resize((256,256))
dt_wm=Image.open(pick['trigger_path']).convert("RGB").resize((256,256))
clean_a=np.asarray(clean,np.float32)

def resid(wm):
    return np.abs(np.asarray(wm,np.float32)-clean_a)

# --- peer encoders, each in its own process ---
# LIDMark, LampMark and StegaStamp all ship a top-level package called "model",
# so they cannot be imported into one interpreter; every peer goes through the
# same isolated helper for consistency.
import subprocess, tempfile
_TMP = tempfile.mkdtemp(prefix="fig4_")
clean.save(f"{_TMP}/clean.png")

def peer(name):
    out = f"{_TMP}/{name}.png"
    script = "eg_embed_one.py" if name == "editguard" else "embed_one.py"
    args = ([f"{_TMP}/clean.png", out, "256"] if name == "editguard"
            else [name, f"{_TMP}/clean.png", out])
    cwd = _dt("DT_PEERS_ROOT", "/EditGuard/code") if name == "editguard" else None
    subprocess.run([sys.executable,
                    _dt("DT_PEERS_ROOT") + f"/{script}"] + args,
                   check=True, capture_output=True, cwd=cwd)
    return Image.open(out).convert("RGB").resize((256, 256))

def psnr(wm):
    m=np.mean((np.asarray(wm,np.float32)-clean_a)**2); return 100 if m<=1e-8 else 20*np.log10(255)-10*np.log10(m)

methods=[("Ours (phase)",dt_wm,"#2a7"),("FaceSigns",peer("facesigns"),"#000"),
         ("EditGuard",peer("editguard"),"#000"),("SepMark",peer("sepmark"),"#000"),
         ("LIDMark",peer("lidmark"),"#000"),("StegaStamp",peer("stegastamp"),"#000"),
         ("KAD-Net",peer("kadnet"),"#000"),("LampMark",peer("lampmark"),"#000")]

AMP=10.0
ncol=len(methods)
fig,ax=plt.subplots(2,ncol,figsize=(2.3*ncol,4.3),gridspec_kw={"wspace":0.28,"hspace":0.06})
def frame(a):
    a.add_patch(plt.Rectangle((0,0),1,1,transform=a.transAxes,fill=False,
                              edgecolor="black",linewidth=1.4,clip_on=False))
for k,(name,img,col) in enumerate(methods):
    ax[0,k].imshow(img); ax[0,k].axis("off"); frame(ax[0,k])
    ax[0,k].set_title(f"{name}\n{psnr(img):.1f} dB",fontsize=26,color=col,
                      fontweight="bold" if name.startswith("Ours") else "normal")
    ax[1,k].imshow(np.clip(resid(img)*AMP,0,255).astype("uint8")); ax[1,k].axis("off"); frame(ax[1,k])
plt.tight_layout(pad=0.4)
for e in ("pdf","png"): plt.savefig(f"{OUT}/fig_imperceptibility.{e}",dpi=150,bbox_inches="tight")
print("saved",OUT+"/fig_imperceptibility.pdf")
print("PSNRs:", {n:round(psnr(w),1) for n,w,_ in methods})
