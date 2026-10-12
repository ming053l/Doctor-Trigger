"""Fig 5: visual fidelity of released images.

Top: one face as released by our enrollment and as marked by each peer method, with PSNR against
the original; bottom: 10x-amplified residuals. Our panel is an earlier enrollment render (an
original crop and its release, listed by JSON sidecars matched by ``legacy_trigger_glob``); every
peer marks the same original crop.

Each peer runs in its own process through the embed helpers in ``peers_root`` (LIDMark, LampMark and
StegaStamp all ship a top-level package called ``model``, so they cannot share one interpreter).

Paths from config.yaml (doctor_trigger.config): legacy_trigger_glob, peers_root, outputs_root.
Writes <outputs_root>/figures/fig_imperceptibility.{pdf,png}.
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"]="Times New Roman"
matplotlib.rcParams["mathtext.fontset"]="stix"
import matplotlib.pyplot as plt
from PIL import Image

from doctor_trigger import config

OUT = config.path("outputs_root") + "/figures"
PEERS_ROOT = config.path("peers_root")
os.makedirs(OUT, exist_ok=True)

# the first enrollment render whose original crop and release both exist
pick = None
for sidecar in sorted(glob.glob(config.path("legacy_trigger_glob"), recursive=True)):
    entry = json.load(open(sidecar))
    if os.path.exists(entry['source']) and os.path.exists(entry['trigger_path']):
        pick = entry; break
original = Image.open(pick['source']).convert("RGB").resize((256,256))
released = Image.open(pick['trigger_path']).convert("RGB").resize((256,256))
original_a = np.asarray(original, np.float32)

def residual(marked):
    return np.abs(np.asarray(marked, np.float32) - original_a)

WORK = tempfile.mkdtemp(prefix="fig_imperceptibility_")
original.save(f"{WORK}/clean.png")

def peer(name):
    """Mark the original crop with one peer method (in its own process); returns the marked image."""
    out = f"{WORK}/{name}.png"
    script = "eg_embed_one.py" if name == "editguard" else "embed_one.py"
    args = ([f"{WORK}/clean.png", out, "256"] if name == "editguard"
            else [name, f"{WORK}/clean.png", out])
    cwd = PEERS_ROOT + "/EditGuard/code" if name == "editguard" else None
    subprocess.run([sys.executable,
                    PEERS_ROOT + f"/{script}"] + args,
                   check=True, capture_output=True, cwd=cwd)
    return Image.open(out).convert("RGB").resize((256, 256))

def psnr(marked):
    m=np.mean((np.asarray(marked,np.float32)-original_a)**2); return 100 if m<=1e-8 else 20*np.log10(255)-10*np.log10(m)

methods=[("Ours (phase)",released,"#2a7"),("FaceSigns",peer("facesigns"),"#000"),
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
    ax[1,k].imshow(np.clip(residual(img)*AMP,0,255).astype("uint8")); ax[1,k].axis("off"); frame(ax[1,k])
plt.tight_layout(pad=0.4)
for e in ("pdf","png"): plt.savefig(f"{OUT}/fig_imperceptibility.{e}",dpi=150,bbox_inches="tight")
print("saved",OUT+"/fig_imperceptibility.pdf")
print("PSNRs:", {n:round(psnr(w),1) for n,w,_ in methods})
