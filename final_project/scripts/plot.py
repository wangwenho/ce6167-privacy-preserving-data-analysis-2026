import os

import matplotlib.pyplot as plt
import pandas as pd

# ── 設定專業繪圖風格 ──
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# ── 讀取資料 ──
csv_path = "outputs/results.csv"
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"{csv_path} not found. Run batch_eval.py first.")
df = pd.read_csv(csv_path)
required = ["noise", "rouge1", "bleu4", "exact_match", "edit_mean", "embed_sim"]
if not all(c in df.columns for c in required):
    raise ValueError(f"CSV missing columns. Need: {required}")

# ── 處理 X 軸 ──
EPS = 1e-4
x = df["noise"].values.copy()
x[x == 0] = EPS
XTICK_VALS = [EPS if v == 0 else v for v in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]]
XTICK_LABS = ["0", "0.001", "0.005", "0.01", "0.05", "0.1", "0.5"]

os.makedirs("outputs", exist_ok=True)


# ── 通用函數 ──
def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=4)
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)


def add_threshold_region(ax):
    ax.axvspan(0.01, 0.05, alpha=0.08, color="gray")
    ax.axvline(x=0.01, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.axvline(x=0.05, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.text(
        0.022,
        0.94,
        "Threshold\nRegion",
        fontsize=9,
        color="gray",
        ha="center",
        fontweight="bold",
        alpha=0.7,
        transform=ax.get_xaxis_transform(),
    )


def set_log_xaxis(ax):
    ax.set_xscale("log")
    ax.set_xlim([5e-5, 0.8])
    ax.set_xlabel("Gaussian Noise Scale ($\\sigma$)", fontsize=14)
    ax.set_xticks(XTICK_VALS)
    ax.set_xticklabels(XTICK_LABS, rotation=40)


# ═══════════════════════════════════════════════════════
# Figure 1 — Generation Quality
# ═══════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(8, 4.5))
style_axis(ax1)
set_log_xaxis(ax1)
add_threshold_region(ax1)

ax1.plot(
    x,
    df["rouge1"],
    "o-",
    color="#0072B2",
    linewidth=2,
    markersize=6,
    markerfacecolor="white",
    label="ROUGE-1",
)
ax1.plot(
    x,
    df["bleu4"],
    "s-",
    color="#56B4E9",
    linewidth=2,
    markersize=6,
    markerfacecolor="white",
    label="BLEU-4",
)
ax1.plot(
    x,
    df["exact_match"],
    "^-",
    color="#009E73",
    linewidth=2,
    markersize=6,
    markerfacecolor="white",
    label="Exact Match",
)
ax1.set_ylabel("Score", fontsize=14)
ax1.set_ylim([-0.02, 1.02])
ax1.legend(
    loc="upper right", fontsize=10, frameon=True, framealpha=0.9, edgecolor="#CCCCCC"
)
ax1.set_title("Generation Quality", fontsize=14, fontweight="bold", pad=10)

fig1.savefig("outputs/fig1_generation_quality.png", dpi=300)
fig1.savefig("outputs/fig1_generation_quality.pdf")
plt.close(fig1)
print("[OK] outputs/fig1_generation_quality.png + .pdf")

# ═══════════════════════════════════════════════════════
# Figure 2 — Edit Distance
# ═══════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(8, 4.5))
style_axis(ax2)
set_log_xaxis(ax2)
add_threshold_region(ax2)

ax2.plot(
    x,
    df["edit_mean"],
    "o-",
    color="#CC79A7",
    linewidth=2,
    markersize=5,
    markerfacecolor="white",
    label="Edit Distance",
)
ax2.set_ylabel("Edit Distance (mean)", fontsize=14)
ax2.set_ylim([10, 65])
ax2.set_title("Edit Distance", fontsize=14, fontweight="bold", pad=10)

fig2.savefig("outputs/fig2_edit_distance.png", dpi=300)
fig2.savefig("outputs/fig2_edit_distance.pdf")
plt.close(fig2)
print("[OK] outputs/fig2_edit_distance.png + .pdf")

# ═══════════════════════════════════════════════════════
# Figure 3 — Semantic Similarity
# ═══════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(8, 4.5))
style_axis(ax3)
set_log_xaxis(ax3)
add_threshold_region(ax3)

baseline_embed = df[df["noise"] == 0]["embed_sim"].values[0]

ax3.plot(
    x,
    df["embed_sim"],
    "o-",
    color="#E69F00",
    linewidth=2.5,
    markersize=7,
    markerfacecolor="white",
    label="Embedding Similarity",
)
ax3.axhline(
    y=baseline_embed,
    color="#E69F00",
    linewidth=1.0,
    linestyle="--",
    alpha=0.5,
    label=f"Baseline ({baseline_embed:.3f})",
)
ax3.set_ylabel("Embedding Similarity (cosine)", fontsize=14)
ax3.set_ylim([0.55, 0.92])
ax3.legend(
    loc="lower left", fontsize=10, frameon=True, framealpha=0.9, edgecolor="#CCCCCC"
)
ax3.set_title("Semantic Similarity", fontsize=14, fontweight="bold", pad=10)

fig3.savefig("outputs/fig3_embedding_similarity.png", dpi=300)
fig3.savefig("outputs/fig3_embedding_similarity.pdf")
plt.close(fig3)
print("[OK] outputs/fig3_embedding_similarity.png + .pdf")

print("\nDone! Publication-quality figures saved to outputs/.")
