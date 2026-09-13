import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py, or export it (see config.example.yaml)"
    return v + rest
"""C2PA-style deployment round trip on the canonical assets.

Each released asset carries a signed credential (HMAC-SHA256 manifest over the
asset id and the record hash, embedded as PNG tEXt / JPEG COM). The asset is
pushed through a platform-like channel, comes back, and the verifier (1) tries
to resolve the credential and (2) scores the face against the private record at
the tau calibrated to 1% FPR on the canonical FF++ train benign split.
Both the genuine release and its recorded SimSwap re-rendering make the trip.
"""
import io, json, hmac, hashlib, sys
import numpy as np, pandas as pd
from PIL import Image
from PIL.PngImagePlugin import PngInfo
sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from scripts.benign_channels import transform_image

R = _dt("DT_CANON_ROOT")
TAU = 0.9821                       # 1% FPR on FF++ train benign (canonical features)
KEY = b"platform-signing-key-demo"
N = 200
EPS = 1e-12

cases = pd.read_csv(f"{R}/common/cases.csv")
cases = cases[(cases.split == "test") & (cases.dataset == "FF++")]
enr = cases.groupby("enrollment_id").first().head(N)

def manifest_for(row):
    body = json.dumps({"asset_id": row.Index,
                       "record_sha256": row.doctor_trigger_template_sha256[:16]})
    sig = hmac.new(KEY, body.encode(), hashlib.sha256).hexdigest()[:32]
    return json.dumps({"m": body, "sig": sig})

def attach(img, manifest, fmt, quality=None, keep_meta=True):
    b = io.BytesIO()
    if fmt == "PNG":
        if keep_meta:
            pi = PngInfo(); pi.add_text("c2pa_manifest", manifest)
            img.save(b, "PNG", pnginfo=pi)
        else:
            img.save(b, "PNG")
    else:
        kw = {"quality": quality or 85}
        if keep_meta: kw["comment"] = manifest.encode()
        img.save(b, "JPEG", **kw)
    b.seek(0); return Image.open(b)

def resolve(img, row):
    raw = img.text.get("c2pa_manifest") if hasattr(img, "text") and img.text else None
    if raw is None:
        c = img.info.get("comment")
        raw = c.decode() if isinstance(c, bytes) else c
    if not raw: return False
    try:
        d = json.loads(raw)
        ok = hmac.new(KEY, d["m"].encode(), hashlib.sha256).hexdigest()[:32] == d["sig"]
        return ok and json.loads(d["m"])["asset_id"] == row.Index
    except Exception:
        return False

def phase_of(img):
    rgb = np.asarray(img.convert("RGB"), np.float64)
    lum = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
    return np.fft.fftshift(np.angle(np.fft.fft2(lum)))

def H_of(img, tp):
    with np.load(tp) as rec:
        host = rec["host_phase"].astype(np.float64); sup = rec["support"].astype(bool)
        w = rec["host_amplitude"].astype(np.float64)
    if (img.size[1], img.size[0]) != host.shape:
        img = img.resize((host.shape[1], host.shape[0]), Image.BICUBIC)
    r = np.cos((phase_of(img) - host + np.pi) % (2*np.pi) - np.pi)
    _cut = float(_dtos.environ.get("DT_KAPPA_CUT", "0"))
    if _cut > 0:
        fy = np.fft.fftshift(np.fft.fftfreq(host.shape[0])).reshape(-1, 1)
        fx = np.fft.fftshift(np.fft.fftfreq(host.shape[1])).reshape(1, -1)
        rad = np.sqrt(fy*fy + fx*fx) / np.sqrt(0.5)
        sup = sup & (rad <= _cut)
    ww = np.where(sup, w, 0.0)
    return float(np.sum(r*ww) / (np.sum(ww) + EPS))

# ---- channels: (name, function(img, manifest) -> PIL image as it comes back) ----
def ch_lossless(img, man):   return attach(img, man, "PNG", keep_meta=True)
def ch_jpeg85(img, man):     return attach(img, man, "JPEG", 85, keep_meta=True)
def ch_repost(img, man):
    x = img.resize((int(img.width*0.75), int(img.height*0.75)), Image.BICUBIC)
    return attach(x, man, "JPEG", 70, keep_meta=False)          # platforms strip metadata
def ch_repost2(img, man):
    x = ch_repost(img, man)
    y = x.resize((int(x.width*0.75), int(x.height*0.75)), Image.BICUBIC)
    return attach(y, man, "JPEG", 70, keep_meta=False)
def ch_strip(img, man):      return attach(img, man, "PNG", keep_meta=False)
def ch_screenshot(img, man):
    x = img.resize((int(img.width*1.25), int(img.height*1.25)), Image.BICUBIC)
    return attach(x, man, "PNG", keep_meta=False)
def ch_transcode(img, man):
    x = transform_image(img.convert("RGB"), "x264_crf30", seed=7)
    return attach(x, man, "PNG", keep_meta=False)
def ch_crop(img, man):
    wc, hc = int(img.width*0.9), int(img.height*0.9)
    x = img.crop(((img.width-wc)//2, (img.height-hc)//2, (img.width-wc)//2+wc, (img.height-hc)//2+hc))
    return attach(x, man, "PNG", keep_meta=True)                # credential survives; alignment does not

CHANNELS = [("lossless re-encode", ch_lossless), ("JPEG q85, EXIF-preserving", ch_jpeg85),
            ("social repost (0.75x + JPEG q70)", ch_repost), ("repost of a repost", ch_repost2),
            ("metadata stripping only", ch_strip), ("screenshot (1.25x + PNG)", ch_screenshot),
            ("video transcode (x264 crf30)", ch_transcode), ("center crop 90%", ch_crop)]

rows = []
for name, fn in CHANNELS:
    res_ok = acc = flag = 0; n = 0; hs = []
    for row in enr.itertuples():
        man = manifest_for(row)
        gen = Image.open(cases[(cases.enrollment_id == row.Index) & (cases.protocol == "benign")].input_path.iloc[0])
        swap_img = Image.open(cases[(cases.enrollment_id == row.Index) & (cases.protocol == "swap")].input_path.iloc[0])
        back_g = fn(gen, man); back_t = fn(swap_img, man)
        res_ok += int(resolve(back_g, row))
        hg = H_of(back_g, row.doctor_trigger_template_path)
        ht = H_of(back_t, row.doctor_trigger_template_path)
        hs.append(hg)
        acc += int(hg >= TAU); flag += int(ht < TAU); n += 1
    print(json.dumps({"channel": name, "resolve": res_ok/n, "genuine_preserved": acc/n,
                      "swap_flagged": flag/n, "mean_benign_H": float(np.mean(hs)), "n": n}))

# ---- second pass: declared-channel calibration ----
# a deployment that declares a channel calibrates tau on genuine train copies
# pushed through that channel; test-side status is then measured at that tau.
train = pd.read_csv(f"{R}/common/cases.csv")
train = train[(train.split == "train") & (train.dataset == "FF++") & (train.protocol == "benign")]
tr = train.groupby("enrollment_id").first().head(100)
print("---DECLARED---")
for name, fn in CHANNELS:
    ths = []
    for row in tr.itertuples():
        img = Image.open(row.input_path)
        ths.append(H_of(fn(img, manifest_for(row)), row.doctor_trigger_template_path))
    tau_ch = float(np.quantile(ths, 0.01))
    acc = flag = n = 0
    for row in enr.itertuples():
        man = manifest_for(row)
        gen = Image.open(cases[(cases.enrollment_id == row.Index) & (cases.protocol == "benign")].input_path.iloc[0])
        swap_img = Image.open(cases[(cases.enrollment_id == row.Index) & (cases.protocol == "swap")].input_path.iloc[0])
        acc += int(H_of(fn(gen, man), row.doctor_trigger_template_path) >= tau_ch)
        flag += int(H_of(fn(swap_img, man), row.doctor_trigger_template_path) < tau_ch)
        n += 1
    print(json.dumps({"channel": name, "tau_declared": tau_ch,
                      "genuine_preserved": acc/n, "swap_flagged": flag/n, "n": n}))
