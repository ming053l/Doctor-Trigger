"""Peer rows of Table 2: each watermarking peer run end to end on the test split.

For every enrollment the peer embeds its own mark into ``enrollment_input_path``; the marked image
goes through a PNG round trip (as the benign queries do) and is scored as the benign query, and
each recorded SimSwap attack (``attack_source_path``) is applied to it and scored as the swap
query. One row per query_id; an embedding, swap or scoring failure is recorded as "quarantine"
with its exception name rather than replaced.

Usage: peer_detection.py METHOD [--dataset FF++|DFDC|CelebDF] [--engine ENGINE] [--pin-checkpoint TAG]
  --dataset         one test dataset only (default: all three)
  --engine          face re-rendering engine of experiments/shared/swap_engines.py (default simswap)
  --pin-checkpoint  use the peer checkpoint trained on TAG (ffpp | dfdc | celebdf) for every
                    dataset (zero-shot transfer); by default each dataset uses its own
Writes <outputs_root>/<METHOD>[_pin_<TAG>][_<ENGINE>].csv with columns query_id, method_id,
score_fake, status, error; method_id is "<METHOD>_canonical_v2", the identifier the result
tables were built with.
"""
import argparse
import csv
import tempfile
from pathlib import Path

from PIL import Image

from doctor_trigger import config
from doctor_trigger.manifests import DATASET_TAGS, read_cases
from experiments.peer_methods import load_peer
from experiments.shared.swap_engines import ENGINES

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("method")
parser.add_argument("--dataset")
parser.add_argument("--engine", default="simswap")
parser.add_argument("--pin-checkpoint", default="")
args = parser.parse_args()

cases = read_cases(split="test").copy()
if args.dataset:
    cases = cases[cases.dataset == args.dataset].copy()
swap = ENGINES[args.engine]()
suffix = (f"_pin_{args.pin_checkpoint}" if args.pin_checkpoint else "") + \
         (f"_{args.engine}" if args.engine != "simswap" else "")
out_path = Path(config.path("outputs_root")) / f"{args.method}{suffix}.csv"

rows, quarantined = [], 0
with tempfile.TemporaryDirectory(prefix=f"peer_{args.method}_") as scratch:
    for dataset, dataset_cases in cases.groupby("dataset"):
        embed, score = load_peer(args.method, args.pin_checkpoint or DATASET_TAGS[dataset])
        for enrollment_id, group in dataset_cases.groupby("enrollment_id"):
            benign = group[group.protocol == "benign"]
            swaps = group[group.protocol == "swap"]
            try:
                marked = embed(Image.open(group.enrollment_input_path.iloc[0]).convert("RGB"))
                marked_path = f"{scratch}/{enrollment_id}.png"
                marked.save(marked_path)
                marked = Image.open(marked_path).convert("RGB")
            except Exception as error:
                for _, query in group.iterrows():
                    rows.append((query.query_id, "quarantine", "", f"embed:{type(error).__name__}")); quarantined += 1
                continue
            for _, query in benign.iterrows():
                try:
                    rows.append((query.query_id, "pass", score(marked), ""))
                except Exception as error:
                    rows.append((query.query_id, "quarantine", "", f"score:{type(error).__name__}")); quarantined += 1
            for _, query in swaps.iterrows():
                try:
                    swapped = swap(marked, Image.open(query.attack_source_path).convert("RGB")).resize(marked.size)
                    rows.append((query.query_id, "pass", score(swapped), ""))
                except Exception as error:
                    rows.append((query.query_id, "quarantine", "", f"swap:{type(error).__name__}")); quarantined += 1

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["query_id", "method_id", "score_fake", "status", "error"])
    for query_id, status, score_fake, error in rows:
        writer.writerow([query_id, f"{args.method}_canonical_v2", score_fake, status, error])
print(f"{args.method}: {len(rows)} rows, {quarantined} quarantined -> {out_path}")
