"""
Qualitative comparison of reconstruction quality at different noise scales.

Scans log files in models/, extracts one sentence across all noise levels,
and produces a markdown table (stdout) plus a styled matplotlib table (PNG).
"""

import argparse
import glob
import json
import os
import re
import sys

import evaluate
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-ticks")
plt.rcParams.update(
    {
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

LOG_DIR = "models"
BASELINE_PATTERN = "attacker_gpt2_large_personachat_mpnet_beam_noise_0.0.log"
NOISE_PATTERN = "attacker_gpt2_large_personachat_mpnet_beam_noise_*.log"
OUTPUT_DIR = "outputs"

_rouge_evaluator = None


def _get_rouge():
    global _rouge_evaluator
    if _rouge_evaluator is None:
        _rouge_evaluator = evaluate.load("rouge")
    return _rouge_evaluator


def extract_noise(fname):
    """Parse noise scale from log filename."""
    m = re.search(r"_noise_([\d.]+)\.log$", fname)
    if m:
        return float(m.group(1))
    if "noise" not in fname:
        return 0.0  # baseline (no _noise_ in name)
    return None


def gather_logs(log_dir):
    """Return list of (noise, path) tuples sorted by noise ascending."""
    entries = []
    base_path = os.path.join(log_dir, BASELINE_PATTERN)
    if os.path.isfile(base_path):
        entries.append((0.0, base_path))

    for path in sorted(glob.glob(os.path.join(log_dir, NOISE_PATTERN))):
        fname = os.path.basename(path)
        noise = extract_noise(fname)
        if noise is None:
            continue
        entries.append((noise, path))

    # Deduplicate by noise (prefer first occurrence)
    seen = set()
    unique = []
    for noise, path in entries:
        if noise not in seen:
            seen.add(noise)
            unique.append((noise, path))
    unique.sort(key=lambda x: x[0])
    return unique


def load_log(path):
    """Load a log JSON file and return (gt_list, pred_list)."""
    with open(path) as f:
        data = json.load(f)
    pred = [s.replace("<|endoftext|>", "") for s in data["pred"]]
    return data["gt"], pred


def rouge1_score(reference, prediction):
    """Compute ROUGE-1 F1 score between two strings."""
    rouge = _get_rouge()
    result = rouge.compute(predictions=[prediction], references=[reference])
    return result["rouge1"]


def truncate_text(text, max_words=20):
    """Truncate text to max_words words for display."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + " ..."
    return text


def build_markdown_table(rows, gt_sentence, index):
    """Build a markdown table string from rows."""
    lines = []
    lines.append(f"## Qualitative Comparison (sentence index={index})")
    lines.append(f"**Ground truth:** {gt_sentence}")
    lines.append("")
    lines.append("| Noise Scale | Reconstructed Sentence |")
    lines.append("|:-----------:|:-----------------------|")
    for noise, sent in rows:
        noise_str = f"{noise:.3f}" if noise > 0 else "0 (baseline)"
        lines.append(f"| {noise_str} | {sent} |")
    return "\n".join(lines)


def build_matplotlib_table(rows, gt_sentence, index, fontsize=10):
    """Render a styled matplotlib table and save as PNG."""
    nrows = len(rows)
    fig_height = nrows * 0.4
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")

    cell_text = []
    for noise, sent in rows:
        noise_str = f"{noise:.3f}" if noise > 0 else "0 (baseline)"
        cell_text.append([noise_str, sent])

    col_labels = [
        "Noise Scale ($\\sigma$)",
        "Reconstructed Sentence",
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
    )

    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.6)  # increase row height for readability

    col_widths = [0.15, 0.85]
    for i, width in enumerate(col_widths):
        for row_idx in range(nrows + 1):
            cell = table[row_idx, i]
            cell.set_width(width)
            if row_idx == 0:
                cell.set_text_props(weight="regular", fontsize=fontsize)
                cell.set_facecolor("#404040")
                cell.set_text_props(color="white", weight="regular", fontsize=fontsize)
            elif i == 0:
                cell.set_text_props(
                    ha="center",
                    fontfamily="monospace",
                    weight="regular",
                    fontsize=fontsize,
                )
            else:
                cell.set_text_props(ha="left", fontsize=fontsize)

    for row_idx in range(1, nrows + 1):
        for col_idx in range(2):
            cell = table[row_idx, col_idx]
            if row_idx % 2 == 0:
                cell.set_facecolor("#f8f8f8")
            else:
                cell.set_facecolor("white")

    gt_row = 1
    for col_idx in range(2):
        table[gt_row, col_idx].set_facecolor("#e8e8e8")

    for row_idx in range(nrows + 1):
        for col_idx in range(2):
            cell = table[row_idx, col_idx]
            cell.set_edgecolor("#cccccc")
            cell.set_linewidth(0.5)
            cell.set_text_props(va="center")

    ax.set_title(
        "Reconstruction Examples Across Noise Scales",
        fontsize=10,
        fontweight="bold",
        pad=6,
    )

    gt_display = truncate_text(gt_sentence, max_words=25)
    ax.text(
        0.5,
        0.98,
        f'Ground truth index {index}: "{gt_display}"',
        ha="center",
        va="bottom",
        fontsize=8,
        style="italic",
        color="#555555",
        transform=ax.transAxes,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    png_path = os.path.join(OUTPUT_DIR, "fig4_reconstruction_examples.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return png_path


def main():
    parser = argparse.ArgumentParser(
        description="Show how the same input sentence is reconstructed at different noise levels."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=5,
        help="Index of the sentence to compare (default: 5).",
    )
    parser.add_argument(
        "--log-dir",
        default=LOG_DIR,
        help="Directory containing log files (default: models/).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for output figures (default: outputs/).",
    )
    args = parser.parse_args()

    log_entries = gather_logs(args.log_dir)
    if not log_entries:
        print(f"ERROR: No log files found matching patterns in {args.log_dir}/")
        sys.exit(1)

    print(f"Found {len(log_entries)} log file(s):")
    for noise, path in log_entries:
        print(f"  [{noise:>7.4f}] {os.path.basename(path)}")
    print()

    _, baseline_path = log_entries[0]
    gt_all, _ = load_log(baseline_path)

    if args.index < 0 or args.index >= len(gt_all):
        print(
            f"ERROR: Index {args.index} out of range (0-{len(gt_all) - 1}). "
            f"Log has {len(gt_all)} sentences."
        )
        sys.exit(1)

    sentence_index = args.index
    gt_sentence = gt_all[sentence_index]
    word_count = len(gt_sentence.split())

    print(f"Selected sentence index: {sentence_index}")
    print(f"Word count: {word_count}")
    print(f"Ground truth: {gt_sentence}")
    print()

    rows = []
    for noise, path in log_entries:
        _, pred_list = load_log(path)
        if sentence_index >= len(pred_list):
            print(
                f"  WARNING: {os.path.basename(path)} has only {len(pred_list)} predictions, "
                f"skipping index {sentence_index}."
            )
            continue
        predicted = pred_list[sentence_index]
        rows.append((noise, predicted))

    if not rows:
        print("ERROR: No valid predictions extracted.")
        sys.exit(1)

    md_table = build_markdown_table(
        [(n, truncate_text(s, 20)) for n, s in rows], gt_sentence, sentence_index
    )
    print(md_table)
    print()

    png_path = build_matplotlib_table(rows, gt_sentence, sentence_index)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
