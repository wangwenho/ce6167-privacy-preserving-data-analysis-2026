"""
Qualitative comparison of reconstruction quality at different noise scales.

Scans log files in models/, extracts one sentence across all noise levels,
and produces a markdown table (stdout) plus a styled matplotlib table (PNG + PDF).
"""

import argparse
import glob
import json
import os
import re
import sys

import evaluate
import matplotlib.pyplot as plt

# ── Log file pattern ──
LOG_DIR = "models"
BASELINE_PATTERN = "attacker_gpt2_large_personachat_mpnet_beam.log"
NOISE_PATTERN = "attacker_gpt2_large_personachat_mpnet_beam_noise_*.log"
OUTPUT_DIR = "outputs"


def extract_noise(fname):
    """Parse noise scale from log filename. Mirrors batch_eval.py logic."""
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
    rouge = evaluate.load("rouge")
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


def build_matplotlib_table(rows, gt_sentence, index, threshold_row_idx):
    """Render a styled matplotlib table and save as PNG + PDF."""
    nrows = len(rows)
    fig_height = nrows * 0.4
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")

    # Prepare cell text
    cell_text = []
    for noise, sent in rows:
        noise_str = f"{noise:.3f}" if noise > 0 else "0 (baseline)"
        cell_text.append([noise_str, sent])

    col_labels = ["Noise Scale", "Reconstructed Sentence"]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
    )

    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)  # increase row height for readability

    col_widths = [0.15, 0.85]
    for i, width in enumerate(col_widths):
        for row_idx in range(nrows + 1):  # +1 for header
            cell = table[row_idx, i]
            cell.set_width(width)
            if row_idx == 0:
                # Header
                cell.set_text_props(weight="bold", fontsize=10)
                cell.set_facecolor("#404040")
                cell.set_text_props(color="white", weight="bold", fontsize=10)
            elif i == 0:
                # Noise column: centered, monospace, bold
                cell.set_text_props(
                    ha="center", fontfamily="monospace", weight="bold", fontsize=10
                )
            else:
                # Sentence column: left-aligned, regular
                cell.set_text_props(ha="left", fontsize=10)

    # Alternating row colors
    for row_idx in range(1, nrows + 1):
        for col_idx in range(2):
            cell = table[row_idx, col_idx]
            if row_idx % 2 == 0:
                cell.set_facecolor("#f5f5f5")
            else:
                cell.set_facecolor("white")

    # Highlight ground truth row (index 1 = first data row for baseline noise=0)
    # Baseline is row index 1 in the table (0 is header)
    gt_row = 1  # baseline is always first
    for col_idx in range(2):
        table[gt_row, col_idx].set_facecolor("#e8e8e8")

    # Cell padding (edge text color)
    for row_idx in range(nrows + 1):
        for col_idx in range(2):
            cell = table[row_idx, col_idx]
            cell.set_edgecolor("#cccccc")
            cell.set_linewidth(0.5)
            # vertical alignment
            cell.set_text_props(va="center")

    # Title
    ax.set_title(
        "Qualitative Comparison of Reconstruction Quality",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # Subtitle with ground truth
    gt_display = truncate_text(gt_sentence, max_words=25)
    fig.text(
        0.5,
        0.94,
        f'Ground truth [index {index}]: "{gt_display}"',
        ha="center",
        fontsize=9,
        style="italic",
        color="#555555",
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    png_path = os.path.join(OUTPUT_DIR, "fig4_qualitative_comparison.png")
    pdf_path = os.path.join(OUTPUT_DIR, "fig4_qualitative_comparison.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


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

    # ── Gather logs ──
    log_entries = gather_logs(args.log_dir)
    if not log_entries:
        print(f"ERROR: No log files found matching patterns in {args.log_dir}/")
        sys.exit(1)

    print(f"Found {len(log_entries)} log file(s):")
    for noise, path in log_entries:
        print(f"  [{noise:>7.4f}] {os.path.basename(path)}")
    print()

    # ── Load baseline to pick sentence and validate index ──
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

    # ── Extract predicted sentences across noise levels ──
    rows = []  # list of (noise, predicted_sentence)
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

    # ── Compute ROUGE-1 per row to find threshold ──
    rouge_scores = []
    for noise, predicted in rows:
        r1 = rouge1_score(gt_sentence, predicted)
        rouge_scores.append(r1)

    threshold_row_idx = None
    for i, r1 in enumerate(rouge_scores):
        if r1 < 0.3:
            threshold_row_idx = i
            break

    # ── Markdown table ──
    md_table = build_markdown_table(
        [(n, truncate_text(s, 15)) for n, s in rows], gt_sentence, sentence_index
    )
    print(md_table)
    print()

    # ── Matplotlib table ──
    png_path, pdf_path = build_matplotlib_table(
        rows, gt_sentence, sentence_index, threshold_row_idx
    )
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
