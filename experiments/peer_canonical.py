import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py, or export it (see config.example.yaml)"
    return v + rest
"""End-to-end peer run on the canonical protocol (test split).

Per peer_score_contract.json: enroll enrollment_input_path with the method's own
mark, apply the recorded SimSwap attack from attack_source_path, one score row
per query_id, quarantined failures reported rather than replaced.
Usage: peer_canonical.py <method>
"""
import sys, os, csv, tempfile
import numpy as np, pandas as pd, torch
from PIL import Image
sys.path.insert(0, _dt("DT_PEERS_ROOT"))

METHOD = sys.argv[1]
sys.argv = ["x", METHOD, "ffpp"]
import peer_robust
from engines_api import ENGINES

R = _dt("DT_CANON_ROOT")
cases = pd.read_csv(f"{R}/common/cases.csv")
cases = cases[cases.split == "test"].copy()
only_ds = os.environ.get("CANON_DATASET")
if only_ds: cases = cases[cases.dataset == only_ds].copy()
DSMAP = {"FF++": "ffpp", "DFDC": "dfdc", "CelebDF": "celebdf"}
ENGINE = os.environ.get("CANON_ENGINE", "simswap")
swap = ENGINES[ENGINE]()
TAG = os.environ.get("CANON_PIN_DS", "")
ETAG = os.environ.get("CANON_ENGINE", "")
OUT = _dt("DT_OUTPUTS") + f"/{METHOD}{('_pin_'+TAG) if TAG else ''}{('_'+ETAG) if ETAG and ETAG != 'simswap' else ''}.csv"

rows, nq = [], 0
with tempfile.TemporaryDirectory(prefix=f"canon_{METHOD}_") as TMP:
    for ds, grp in cases.groupby("dataset"):
        pin = os.environ.get("CANON_PIN_DS")   # fixed checkpoint for zero-shot transfer runs
        peer_robust.DS = pin if pin else DSMAP[ds]
        embed, score = peer_robust.load()
        for eid, g in grp.groupby("enrollment_id"):
            ben = g[g.protocol == "benign"]; swaps = g[g.protocol == "swap"]
            try:
                wm = embed(Image.open(g.enrollment_input_path.iloc[0]).convert("RGB"))
                wp = f"{TMP}/{eid}.png"; wm.save(wp)
                wm = Image.open(wp).convert("RGB")   # PNG round trip, as the canonical benign query
            except Exception as e:
                for _, q in g.iterrows():
                    rows.append((q.query_id, "quarantine", "", f"embed:{type(e).__name__}")); nq += 1
                continue
            for _, q in ben.iterrows():
                try: rows.append((q.query_id, "pass", score(wm), ""))
                except Exception as e: rows.append((q.query_id, "quarantine", "", f"score:{type(e).__name__}")); nq += 1
            for _, q in swaps.iterrows():
                try:
                    sw = swap(wm, Image.open(q.attack_source_path).convert("RGB")).resize(wm.size)
                    rows.append((q.query_id, "pass", score(sw), ""))
                except Exception as e:
                    rows.append((q.query_id, "quarantine", "", f"swap:{type(e).__name__}")); nq += 1

with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["query_id", "method_id", "score_fake", "status", "error"])
    for qid, st, sc, err in rows:
        w.writerow([qid, f"{METHOD}_canonical_v2", sc, st, err])
print(f"{METHOD}: {len(rows)} rows, {nq} quarantined -> {OUT}")
