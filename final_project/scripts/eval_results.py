import argparse
import csv
import glob
import json
import os
import re
import string
import sys

import editdistance
import evaluate
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer, util

print("Loading evaluation models...")
device = "cuda"
sim_model = SentenceTransformer("sentence-t5-xxl").to(device)
rouge = evaluate.load("rouge")
print("Models loaded.\n")


def extract_noise(fname):
    m = re.search(r"_noise_([\d.]+)\.log$", fname)
    if m:
        return float(m.group(1))
    if "noise" not in fname:
        return 0.0  # baseline
    return None


def evaluate_one(log_path, encode_batch_size=16):
    with open(log_path) as f:
        data = json.load(f)
    gt, pred = data["gt"], data["pred"]

    # Remove special tokens if present
    pred = [s.replace("<|endoftext|>", "") for s in pred]

    # ROUGE
    r = rouge.compute(predictions=pred, references=gt)

    # BLEU
    cands = [s.split() for s in pred]
    refs = [[s.split()] for s in gt]
    bleu1 = nltk.translate.bleu_score.corpus_bleu(refs, cands, weights=(1, 0, 0, 0))
    bleu2 = nltk.translate.bleu_score.corpus_bleu(refs, cands, weights=(0.5, 0.5, 0, 0))
    bleu4 = nltk.translate.bleu_score.corpus_bleu(refs, cands)

    # Exact Match
    def remove_punct(sents):
        out = []
        for s in sents:
            words = [
                w.strip(string.punctuation)
                for w in s.split()
                if w.strip(string.punctuation)
            ]
            out.append(" ".join(words))
        return out

    gt_clean = remove_punct(gt)
    pred_clean = remove_punct(pred)
    em = sum(1 for g, p in zip(gt, pred) if g == p) / len(gt)
    em_nopunct = sum(1 for g, p in zip(gt_clean, pred_clean) if g == p) / len(gt)

    # Edit Distance
    dists = [editdistance.distance(g, p) for g, p in zip(gt, pred)]
    ed_mean = float(np.mean(dists))
    ed_median = float(np.median(dists))

    # Embedding Similarity
    emb_gt = sim_model.encode(gt, convert_to_tensor=True, batch_size=encode_batch_size)
    emb_pred = sim_model.encode(
        pred, convert_to_tensor=True, batch_size=encode_batch_size
    )
    cos_sim = util.cos_sim(emb_gt, emb_pred).diagonal().cpu().numpy()
    embed_sim = float(np.mean(cos_sim))

    return {
        "rouge1": r["rouge1"],
        "rouge2": r["rouge2"],
        "rougeL": r["rougeL"],
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu4": bleu4,
        "exact_match": round(em, 6),
        "exact_match_no_punct": round(em_nopunct, 6),
        "edit_mean": round(ed_mean, 4),
        "edit_median": round(ed_median, 4),
        "embed_sim": round(embed_sim, 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="outputs/results.csv")
    parser.add_argument("--log-dir", default="models")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    pattern = os.path.join(
        args.log_dir, "attacker_gpt2_large_personachat_mpnet_beam*.log"
    )
    log_files = sorted(glob.glob(pattern))
    if not log_files:
        print(f"No logs found: {pattern}")
        sys.exit(1)

    rows = []
    for path in log_files:
        fname = os.path.basename(path)
        noise = extract_noise(fname)
        if noise is None:
            continue
        print(f"  [{noise:>7.4f}] {fname}")
        metrics = evaluate_one(path, encode_batch_size=args.batch_size)
        metrics["noise"] = noise
        rows.append(metrics)

    rows.sort(key=lambda r: r["noise"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    fieldnames = [
        "noise",
        "rouge1",
        "rouge2",
        "rougeL",
        "bleu1",
        "bleu2",
        "bleu4",
        "exact_match",
        "exact_match_no_punct",
        "edit_mean",
        "edit_median",
        "embed_sim",
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. → {len(rows)} results saved to {args.output}")


if __name__ == "__main__":
    main()
