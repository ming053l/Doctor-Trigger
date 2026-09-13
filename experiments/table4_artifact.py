#!/usr/bin/env python
"""Table 4 and Fig. 7: one frozen evidence package with row-level lineage.

An audit found Table 4 assembled from three scripts with different cohorts, units
and statistics, and Fig. 7 drawn from a fourth run. This script recomputes every Table 4 row under
the protocol that produced it, records that protocol row by row, and writes the per-query scores
Fig. 7 is drawn from, so the table and the figure come from one artifact.

  python table4_artifact.py all          # still + clips + collect
  python table4_artifact.py still | clips | collect

Protocols:
  still-image rows   Sub-band reading, kappa cut 0.15. FF++ test, first 200
                     enrollments (groupby order); H_sub = amplitude-weighted cosine of query phase
                     against the HOST phase on the support, radius/sqrt(0.5) <= 0.15; declared tau =
                     1% quantile of H_sub over 100 FF++ train benign enrollments (groupby order)
                     sent through the same channel. G_release_sub (Fig. 7 y axis) = signed
                     projection divided by the released copy's own projection, same sub-band.
  crop 90%           Forward-modeled masked spectrum over the full support,
                     +-2 px placement search; calibration = 100 train benign (drop_duplicates order).
  H.264 intra, clip  Every FF++ test frame (benign and SimSwap) coded as a
                     one-frame libx264 video (crf 30, yuv420p, 1 thread) and decoded; C_tmpl_sub =
                     cosine against the TEMPLATE phase, sub-band 0.15; clip score = mean over the
                     video's frames; tau = 1% quantile of train clip means (first 100 videos).
                     H_sub against the host phase is stored alongside.
Env: DT_CANON_ROOT, DT_MODEL_SNAPSHOT, DT_OUTPUTS (exported by run.py).
Debug only: DT_T4_LIMIT=k restricts every cohort to k items; DT_T4_WORKERS=n.
Outputs in $DT_OUTPUTS/table4/: still_queries.csv, still_calibration.csv, clip_frames.csv,
rows.csv, rows.tex, manifest.json.
"""
import hashlib, hmac, io, json, os, platform, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _dt(env):
    v = os.environ.get(env)
    assert v, f"{env} unset: run through run.py"
    return v


R = _dt("DT_CANON_ROOT")
SNAP = _dt("DT_MODEL_SNAPSHOT")
OUT = Path(_dt("DT_OUTPUTS")) / "table4"
sys.path.insert(0, SNAP)
from scripts.benign_channels import transform_image  # noqa: E402

EPS = 1e-12
CUT = 0.15                    # readout band: radius / sqrt(0.5) <= CUT
TAU_CLEAN = 0.9821            # clean-copy cutoff: 1% FPR on FF++ train benign (canonical features)
N_TEST, N_CAL, CLIP_TRAIN_VIDEOS = 200, 100, 100
KEY = b"platform-signing-key-demo"   # demo HMAC key of the metadata simulation, not a C2PA credential
X264 = "ffmpeg -loop 1 -frames:v 1 -c:v libx264 -threads 1 -crf 30 -pix_fmt yuv420p; decode frame 1"
LIMIT = int(os.environ.get("DT_T4_LIMIT", "0"))
WORKERS = int(os.environ.get("DT_T4_WORKERS", "48"))
STILL = ["lossless", "jpeg85", "repost", "repost2", "strip", "screenshot", "transcode", "crop"]

cases = pd.read_csv(f"{R}/common/cases.csv")


# ---------------------------------------------------------------- cohorts
def cohorts():
    te = cases[(cases.split == "test") & (cases.dataset == "FF++")]
    enr = te.groupby("enrollment_id").first().head(N_TEST)
    tr = cases[(cases.split == "train") & (cases.dataset == "FF++") & (cases.protocol == "benign")]
    cal = tr.groupby("enrollment_id").first().head(N_CAL)            # still-image calibration order
    cal_crop = tr.drop_duplicates("enrollment_id").head(N_CAL)        # crop calibration order
    if LIMIT:
        enr, cal, cal_crop = enr.head(LIMIT), cal.head(LIMIT), cal_crop.head(LIMIT)
    return te, enr, cal, cal_crop


def clip_videos():
    d = cases[(cases.dataset == "FF++") & (cases.protocol.isin(["benign", "swap"]))]
    out = {}
    for split in ("train", "test"):
        ds = d[d.split == split]
        vids = sorted(ds.source_group_id.unique())
        if split == "train":
            vids = vids[:CLIP_TRAIN_VIDEOS]
        if LIMIT:
            vids = vids[:LIMIT]
        out[split] = (ds, vids)
    return out


# ---------------------------------------------------------------- metadata simulation (verbatim)
def manifest_for(eid, sha):
    body = json.dumps({"asset_id": eid, "record_sha256": sha[:16]})
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
        if keep_meta:
            kw["comment"] = manifest.encode()
        img.save(b, "JPEG", **kw)
    b.seek(0)
    return Image.open(b)


def resolve(img, eid):
    raw = img.text.get("c2pa_manifest") if hasattr(img, "text") and img.text else None
    if raw is None:
        c = img.info.get("comment")
        raw = c.decode() if isinstance(c, bytes) else c
    if not raw:
        return False
    try:
        d = json.loads(raw)
        ok = hmac.new(KEY, d["m"].encode(), hashlib.sha256).hexdigest()[:32] == d["sig"]
        return ok and json.loads(d["m"])["asset_id"] == eid
    except Exception:
        return False


# ---------------------------------------------------------------- channels (verbatim)
def ch_lossless(img, man):   return attach(img, man, "PNG", keep_meta=True)
def ch_jpeg85(img, man):     return attach(img, man, "JPEG", 85, keep_meta=True)
def ch_strip(img, man):      return attach(img, man, "PNG", keep_meta=False)


def ch_repost(img, man):
    x = img.resize((int(img.width * 0.75), int(img.height * 0.75)), Image.BICUBIC)
    return attach(x, man, "JPEG", 70, keep_meta=False)


def ch_repost2(img, man):
    x = ch_repost(img, man)
    y = x.resize((int(x.width * 0.75), int(x.height * 0.75)), Image.BICUBIC)
    return attach(y, man, "JPEG", 70, keep_meta=False)


def ch_screenshot(img, man):
    x = img.resize((int(img.width * 1.25), int(img.height * 1.25)), Image.BICUBIC)
    return attach(x, man, "PNG", keep_meta=False)


def ch_transcode(img, man):
    x = transform_image(img.convert("RGB"), "x264_crf30", seed=7)
    return attach(x, man, "PNG", keep_meta=False)


def crop90(img):
    wc, hc = int(img.width * 0.9), int(img.height * 0.9)
    return img.crop(((img.width - wc) // 2, (img.height - hc) // 2,
                     (img.width - wc) // 2 + wc, (img.height - hc) // 2 + hc))


def ch_crop(img, man):       return attach(crop90(img), man, "PNG", keep_meta=True)


CH = dict(lossless=ch_lossless, jpeg85=ch_jpeg85, repost=ch_repost, repost2=ch_repost2,
          strip=ch_strip, screenshot=ch_screenshot, transcode=ch_transcode, crop=ch_crop)


# ---------------------------------------------------------------- measurements
def load_rec(tp):
    with np.load(tp) as rec:
        return dict(host=rec["host_phase"].astype(np.float64), tmpl=rec["template_phase"].astype(np.float64),
                    delta=rec["delta"].astype(np.float64), sup=rec["support"].astype(bool),
                    amp=rec["host_amplitude"].astype(np.float64))


def lum_of(img):
    a = np.asarray(img.convert("RGB"), np.float64)
    return 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]


def wrap(v): return (v + np.pi) % (2 * np.pi) - np.pi


def phase_of(img): return np.fft.fftshift(np.angle(np.fft.fft2(lum_of(img))))


def phase_resized(img, shape):
    if (img.size[1], img.size[0]) != shape:
        img = img.resize((shape[1], shape[0]), Image.BICUBIC)
    return phase_of(img)


def band(shape):
    fy = np.fft.fftshift(np.fft.fftfreq(shape[0])).reshape(-1, 1)
    fx = np.fft.fftshift(np.fft.fftfreq(shape[1])).reshape(1, -1)
    return np.sqrt(fy * fy + fx * fx) / np.sqrt(0.5) <= CUT


def h_host_sub(ph, rec):
    ww = np.where(rec["sup"] & band(rec["host"].shape), rec["amp"], 0.0)
    return float(np.sum(np.cos(wrap(ph - rec["host"])) * ww) / (np.sum(ww) + EPS))


def c_tmpl_sub(ph, rec):
    ww = np.where(rec["sup"] & band(rec["tmpl"].shape), rec["amp"], 0.0)
    return float(np.sum(np.cos(wrap(ph - rec["tmpl"])) * ww) / (np.sum(ww) + EPS))


def g_raw_sub(ph, rec):
    s = rec["sup"] & band(rec["host"].shape)
    w, dl, res = rec["amp"][s], rec["delta"][s], wrap(ph - rec["host"])[s]
    return float(np.sum(w * dl * np.sin(res)) / (np.sum(w * dl ** 2) + EPS))


def forward_h(img, rec):
    sup, amp, tpz = rec["sup"], rec["amp"], rec["tmpl"]
    Hh, Wh = sup.shape; lum = lum_of(img); hc, wc = lum.shape
    spat = np.fft.ifft2(np.fft.ifftshift(np.where(sup, amp, 0.0) * np.exp(1j * tpz))).real
    ww = np.where(sup, amp, 0.0); sww = np.sum(ww) + EPS
    oy0, ox0 = (Hh - hc) // 2, (Wh - wc) // 2
    best = -2.0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            oy, ox = oy0 + dy, ox0 + dx
            if oy < 0 or ox < 0 or oy + hc > Hh or ox + wc > Wh:
                continue
            canvas = np.zeros((Hh, Wh)); canvas[oy:oy + hc, ox:ox + wc] = lum - lum.mean()
            mask = np.zeros((Hh, Wh)); mask[oy:oy + hc, ox:ox + wc] = 1.0
            f_meas = np.fft.fftshift(np.fft.fft2(canvas))
            p_rel = np.fft.fftshift(np.fft.fft2(spat * mask))
            best = max(best, float(np.sum(np.cos(wrap(np.angle(f_meas) - np.angle(p_rel))) * ww) / sww))
    return best


def stable_seed(qid): return int(hashlib.sha256(str(qid).encode()).hexdigest()[:8], 16) % 65521


# ---------------------------------------------------------------- worker
def work(t):
    kind = t["kind"]; rec = load_rec(t["tp"])
    if kind == "still":
        shape = rec["host"].shape
        back = CH[t["channel"]](Image.open(t["path"]), manifest_for(t["eid"], t["sha"]))
        ph = phase_resized(back, shape)
        g, ref = g_raw_sub(ph, rec), g_raw_sub(phase_resized(Image.open(t["rel_path"]), shape), rec)
        out = dict(kind=kind, enrollment_id=t["eid"], query_id=t["qid"], protocol=t["protocol"],
                   channel=t["channel"], credential=int(resolve(back, t["eid"])), H_sub=h_host_sub(ph, rec),
                   G_raw_sub=g, G_ref_sub=ref, G_release_sub=g / max(abs(ref), 1e-6))
        if t["channel"] == "crop":
            out["H_forward"] = forward_h(crop90(Image.open(t["path"])), rec)
        return out
    if kind == "cal":
        back = CH[t["channel"]](Image.open(t["path"]), manifest_for(t["eid"], t["sha"]))
        return dict(kind=kind, enrollment_id=t["eid"], channel=t["channel"],
                    H_sub=h_host_sub(phase_resized(back, rec["host"].shape), rec))
    if kind == "cal_crop":
        return dict(kind=kind, enrollment_id=t["eid"], channel="crop",
                    H_forward=forward_h(crop90(Image.open(t["path"])), rec))
    if kind == "clip":
        frame = transform_image(Image.open(t["path"]).convert("RGB"), "x264_crf30", seed=stable_seed(t["qid"]))
        ph = phase_of(frame)
        return dict(kind=kind, split=t["split"], video=t["video"], query_id=t["qid"], protocol=t["protocol"],
                    C_tmpl_sub=c_tmpl_sub(ph, rec), H_sub=h_host_sub(ph, rec))
    raise ValueError(kind)


def run_pool(tasks, label):
    t0 = time.time(); rows = []
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(work, tasks, chunksize=4), 1):
            rows.append(r)
            if i % 500 == 0 or i == len(tasks):
                print(f"[{label}] {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    return rows


# ---------------------------------------------------------------- stages
def stage_still():
    te, enr, cal, cal_crop = cohorts()
    tasks = []
    for eid, row in enr.iterrows():
        b = te[(te.enrollment_id == eid) & (te.protocol == "benign")].iloc[0]
        f = te[(te.enrollment_id == eid) & (te.protocol == "swap")].iloc[0]
        for ch in STILL:
            for prot, q in (("benign", b), ("swap", f)):
                tasks.append(dict(kind="still", eid=eid, qid=q.query_id, protocol=prot, path=q.input_path,
                                  tp=row.doctor_trigger_template_path, sha=row.doctor_trigger_template_sha256,
                                  rel_path=b.input_path, channel=ch))
    for eid, row in cal.iterrows():
        for ch in STILL:
            if ch != "crop":
                tasks.append(dict(kind="cal", eid=eid, path=row.input_path, tp=row.doctor_trigger_template_path,
                                  sha=row.doctor_trigger_template_sha256, channel=ch))
    for _, row in cal_crop.iterrows():
        tasks.append(dict(kind="cal_crop", eid=row.enrollment_id, path=row.input_path,
                          tp=row.doctor_trigger_template_path))
    rows = run_pool(tasks, "still")
    q = pd.DataFrame([r for r in rows if r["kind"] == "still"]).drop(columns="kind")
    c = pd.DataFrame([r for r in rows if r["kind"] != "still"])
    q.sort_values(["channel", "enrollment_id", "protocol"]).to_csv(OUT / "still_queries.csv", index=False)
    c.sort_values(["kind", "channel", "enrollment_id"]).to_csv(OUT / "still_calibration.csv", index=False)


def stage_clips():
    tasks = []
    for split, (ds, vids) in clip_videos().items():
        for v in vids:
            for _, q in ds[ds.source_group_id == v].iterrows():
                tasks.append(dict(kind="clip", split=split, video=v, qid=q.query_id, protocol=q.protocol,
                                  path=q.input_path, tp=q.doctor_trigger_template_path))
    rows = run_pool(tasks, "clips")
    pd.DataFrame(rows).drop(columns="kind").sort_values(["split", "video", "query_id"]).to_csv(
        OUT / "clip_frames.csv", index=False)


def sha256_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha256_ids(ids): return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def pct(x): return f"{100 * x:.1f}".rstrip("0").rstrip(".")


def stage_collect():
    from sklearn.metrics import roc_auc_score
    q = pd.read_csv(OUT / "still_queries.csv"); cal = pd.read_csv(OUT / "still_calibration.csv")
    clips = pd.read_csv(OUT / "clip_frames.csv")

    def auc(neg, pos): return float(roc_auc_score([0] * len(neg) + [1] * len(pos), np.concatenate([-neg, -pos])))

    rows = []
    base = dict(cohort="FF++ test, first 200 enrollments (groupby order)", unit="image pair")
    for ch in STILL:
        g, f = q[(q.channel == ch) & (q.protocol == "benign")], q[(q.channel == ch) & (q.protocol == "swap")]
        if ch == "crop":
            cc = cal[cal.kind == "cal_crop"].H_forward.values
            gs, fs, stat, calsrc = g.H_forward.values, f.H_forward.values, "forward-modeled masked-spectrum cosine, full support, +-2 px", "100 FF++ train benign, drop_duplicates order, cropped"
        else:
            cc = cal[(cal.kind == "cal") & (cal.channel == ch)].H_sub.values
            gs, fs, stat, calsrc = g.H_sub.values, f.H_sub.values, "H_sub: amplitude-weighted cosine vs host phase, band radius/sqrt(0.5)<=0.15", "100 FF++ train benign, groupby order, same channel"
        tau = float(np.quantile(cc, 0.01))
        rows.append(dict(base, row=ch, n_benign=len(gs), n_swap=len(fs), credential_rate=float(g.credential.mean()),
                         clean_cutoff=TAU_CLEAN, clean_preserved=float(np.mean(gs >= TAU_CLEAN)), declared_tau=tau,
                         preserved=float(np.mean(gs >= tau)), flagged=float(np.mean(fs < tau)), auc=auc(gs, fs),
                         statistic=stat, aggregation="none", calibration=calsrc,
                         score_file="still_queries.csv", calibration_file="still_calibration.csv"))
    tr, te = clips[clips.split == "train"], clips[clips.split == "test"]
    for col, tag in (("C_tmpl_sub", "transcode_clip"), ("H_sub", "transcode_clip_host")):
        tau = float(np.quantile(tr[tr.protocol == "benign"].groupby("video")[col].mean(), 0.01))
        gb = te[te.protocol == "benign"].groupby("video")[col].mean().values
        fb = te[te.protocol == "swap"].groupby("video")[col].mean().values
        ftau = float(np.quantile(tr[tr.protocol == "benign"][col], 0.01))
        fg, ff = te[te.protocol == "benign"][col].values, te[te.protocol == "swap"][col].values
        rows.append(dict(row=tag, cohort="every FF++ test frame; 100 first train videos for tau", unit="clip",
                         n_benign=len(gb), n_swap=len(fb), credential_rate=float("nan"), clean_cutoff=float("nan"),
                         clean_preserved=float("nan"), declared_tau=tau, preserved=float(np.mean(gb >= tau)),
                         flagged=float(np.mean(fb < tau)), auc=auc(gb, fb),
                         frame_tau=ftau, frame_preserved=float(np.mean(fg >= ftau)), frame_flagged=float(np.mean(ff < ftau)),
                         statistic=f"{col}: one-frame libx264 crf30 per frame, band 0.15, "
                                   + ("vs template phase" if col == "C_tmpl_sub" else "vs host phase"),
                         aggregation="mean over the video's frames", calibration="1% quantile of train clip means",
                         score_file="clip_frames.csv", calibration_file="clip_frames.csv"))
    rows_df = pd.DataFrame(rows); rows_df.to_csv(OUT / "rows.csv", index=False)

    r = rows_df.set_index("row")
    lab = {"lossless": ("Lossless re-encode", "record intact"),
           "strip": ("Credential stripping", "lookup needs asset identity"),
           "screenshot": (r"Screenshot ($1.25\times$)", "credential lost; band survives"),
           "jpeg85": (r"JPEG $q85$, EXIF kept", "requantization absorbed"),
           "repost": (r"Social repost ($0.75\times$, $q70$)", "credential gone; record readable"),
           "repost2": ("Repost of a repost", "compounding loss: recalibrate per hop"),
           "transcode": ("H.264 intra, per image", "per-image scores overlap"),
           "crop": (r"Center crop $90\%$", r"crop mask forward-modeled; hold beyond $10\%$ cropped")}
    order = ["lossless", "strip", "screenshot", "jpeg85", "repost", "repost2", "transcode", "crop"]
    lines = []
    for k in order:
        x = r.loc[k]; name, why = lab[k]
        cred = "yes" if x.credential_rate >= 0.995 else "no"
        lines.append(f"{name} & {cred} & ${pct(x.clean_preserved)}\\%$ & flagged ${pct(x.flagged)}\\%$ / "
                     f"preserved ${pct(x.preserved)}\\%$ & {why} \\\\")
        if k == "transcode":
            c = r.loc["transcode_clip"]
            lines.append(f"H.264 intra, clip mean ($n{{=}}{int(c.n_benign)}$) & no & -- & flagged ${pct(c.flagged)}\\%$ / "
                         f"preserved ${pct(c.preserved)}\\%$ & aggregate per clip \\\\")
    (OUT / "rows.tex").write_text("% generated by DT_repo/experiments/table4_artifact.py from rows.csv; do not edit by hand\n"
                                  + "\n".join(lines) + "\n")

    te_c, enr, cal_c, cal_crop = cohorts(); vids = clip_videos()
    repo = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
    try:
        ff = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:
        ff = "unavailable"
    import PIL, sklearn
    manifest = dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), command=" ".join(sys.argv), limit=LIMIT,
        script=dict(path=str(Path(__file__).resolve()), sha256=sha256_file(__file__),
                    repo_rev=git("rev-parse", "HEAD"), repo_dirty=bool(git("status", "--porcelain"))),
        inputs=dict(cases_csv=dict(path=f"{R}/common/cases.csv", sha256=sha256_file(f"{R}/common/cases.csv")),
                    transform_source=dict(path=f"{SNAP}/scripts/benign_channels.py",
                                          sha256=sha256_file(f"{SNAP}/scripts/benign_channels.py"))),
        constants=dict(readout_band="radius/sqrt(0.5) <= 0.15", clean_cutoff=TAU_CLEAN, n_test=N_TEST, n_cal=N_CAL,
                       clip_train_videos=CLIP_TRAIN_VIDEOS, crop_search_px=2, threshold_quantile=0.01,
                       x264=X264, x264_seed="unused by the x264 branch of transform_image",
                       metadata="HMAC-SHA256 JSON in PNG tEXt / JPEG COM (demo key), not a C2PA manifest"),
        cohorts=dict(test_enrollments=dict(n=len(enr), sha256=sha256_ids(enr.index), ids=list(map(str, enr.index))),
                     calibration_enrollments=dict(n=len(cal_c), sha256=sha256_ids(cal_c.index), ids=list(map(str, cal_c.index))),
                     crop_calibration_enrollments=dict(n=len(cal_crop), sha256=sha256_ids(cal_crop.enrollment_id),
                                                       ids=list(map(str, cal_crop.enrollment_id))),
                     clip_test_videos=dict(n=len(vids["test"][1]), sha256=sha256_ids(vids["test"][1])),
                     clip_train_videos=dict(n=len(vids["train"][1]), sha256=sha256_ids(vids["train"][1]))),
        rows=json.loads(rows_df.to_json(orient="records")),
        outputs={p.name: sha256_file(p) for p in sorted(OUT.glob("*")) if p.name != "manifest.json"},
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                         pillow=PIL.__version__, sklearn=sklearn.__version__, ffmpeg=ff, host=platform.node()))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(rows_df[["row", "unit", "n_benign", "credential_rate", "clean_preserved", "declared_tau",
                   "preserved", "flagged", "auc"]].to_string(index=False))
    print("\n".join(lines))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("still", "all"): stage_still()
    if step in ("clips", "all"): stage_clips()
    if step in ("collect", "all"): stage_collect()
