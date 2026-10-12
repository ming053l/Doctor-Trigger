"""End-to-end peer run on the canonical protocol (test split).

Per peer_score_contract.json: enroll enrollment_input_path with the method's own
mark, apply the recorded SimSwap attack from attack_source_path, one score row
per query_id, quarantined failures reported rather than replaced.
Usage: peer_canonical.py <method>
Optional env: CANON_DATASET (one dataset only), CANON_ENGINE (swap engine, default simswap),
CANON_PIN_DS (one fixed peer checkpoint for zero-shot transfer runs).
Writes <outputs_root>/<method>[_pin_<ds>][_<engine>].csv.
"""
import sys, os, csv, tempfile
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
from doctor_trigger import config  # noqa: E402
from doctor_trigger.manifests import DATASET_TAGS, read_cases  # noqa: E402
from experiments.shared.swap_engines import ENGINES  # noqa: E402

METHOD = sys.argv[1]
sys.argv = ["x", METHOD, "ffpp"]
import peer_robust  # noqa: E402  (reads sys.argv at import)

cases = read_cases(split="test").copy()
only_ds = os.environ.get("CANON_DATASET")
if only_ds: cases = cases[cases.dataset == only_ds].copy()
DSMAP = DATASET_TAGS
ENGINE = os.environ.get("CANON_ENGINE", "simswap")
swap = ENGINES[ENGINE]()
TAG = os.environ.get("CANON_PIN_DS", "")
ETAG = os.environ.get("CANON_ENGINE", "")
OUT = config.path("outputs_root") + f"/{METHOD}{('_pin_'+TAG) if TAG else ''}{('_'+ETAG) if ETAG and ETAG != 'simswap' else ''}.csv"

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
