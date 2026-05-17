import os

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("outputs/results.csv")

os.makedirs("outputs", exist_ok=True)

# Replace 0 with a small value for log scale plotting, and prepare x-axis labels
EPS = 1e-4
x = df["noise"].values.copy()
x[x == 0] = EPS
x_label = ["0" if v == 0 else str(v) for v in df["noise"].values]

# Figure 1: generation quality metrics (ROUGE-1, BLEU-4, Exact Match) and Edit Distance
fig, ax1 = plt.subplots(figsize=(9, 5.5))

ax1.plot(
    x, df["rouge1"], "o-", color="#E74C3C", linewidth=2, markersize=6, label="ROUGE-1"
)
ax1.plot(
    x, df["bleu4"], "s-", color="#3498DB", linewidth=2, markersize=6, label="BLEU-4"
)
ax1.plot(
    x,
    df["exact_match"],
    "^-",
    color="#2ECC71",
    linewidth=2,
    markersize=6,
    label="Exact Match",
)
ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

ax2 = ax1.twinx()
ax2.plot(
    x,
    df["edit_mean"],
    "D--",
    color="#9B59B6",
    linewidth=2,
    markersize=5,
    alpha=0.7,
    label="Edit Distance (mean)",
)
ax2.set_ylabel("Edit Distance", fontsize=13)

ax1.axvspan(0.01, 0.05, alpha=0.08, color="orange", label="Threshold region")
ax1.axvline(x=0.01, color="orange", linewidth=1, linestyle=":", alpha=0.6)
ax1.axvline(x=0.05, color="orange", linewidth=1, linestyle=":", alpha=0.6)
ax1.text(
    0.022,
    0.92,
    "Threshold\nRegion",
    fontsize=10,
    color="orange",
    ha="center",
    fontweight="bold",
    alpha=0.8,
)

ax1.set_xscale("log")
ax1.set_xlabel("Noise Scale (σ)", fontsize=14)
ax1.set_ylabel("Score", fontsize=13)
ax1.set_xlim([5e-5, 1])
ax1.set_ylim([-0.02, 1.02])
ax1.set_xticks(x)
ax1.set_xticklabels(x_label, rotation=45, fontsize=8)
ax1.tick_params(axis="y", labelsize=11)
ax2.tick_params(axis="y", labelsize=11)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

ax1.grid(True, alpha=0.25)
ax1.set_title(
    "GEIA Attack Quality vs. Gaussian Noise on Embeddings",
    fontsize=14,
    fontweight="bold",
)

plt.tight_layout()
plt.savefig("outputs/fig1_generation_quality.png", dpi=150)

plt.close()
print("[OK] fig1_generation_quality.png")


# Figure 2: embedding similarity (cosine) vs. noise scale
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(x, df["embed_sim"], "o-", color="#E67E22", linewidth=2.5, markersize=7)
ax.fill_between(x, 0.6, df["embed_sim"], alpha=0.1, color="#E67E22")

ax.axvspan(0.01, 0.05, alpha=0.08, color="orange")
ax.axvline(x=0.01, color="orange", linewidth=1, linestyle=":", alpha=0.6)
ax.axvline(x=0.05, color="orange", linewidth=1, linestyle=":", alpha=0.6)
ax.text(
    0.022,
    0.72,
    "Threshold\nRegion",
    fontsize=10,
    color="orange",
    ha="center",
    fontweight="bold",
    alpha=0.8,
)

ax.set_xscale("log")
ax.set_xlabel("Noise Scale (σ)", fontsize=14)
ax.set_ylabel("Embedding Similarity (cosine)", fontsize=13)
ax.set_xlim([5e-5, 1])
ax.set_ylim([0.6, 0.92])
ax.set_xticks(x)
ax.set_xticklabels(x_label, rotation=45, fontsize=8)
ax.tick_params(axis="y", labelsize=11)
ax.grid(True, alpha=0.25)
ax.set_title(
    "Semantic Similarity (sentence-t5-xxl) vs. Noise", fontsize=14, fontweight="bold"
)

plt.tight_layout()
plt.savefig("outputs/fig2_embedding_similarity.png", dpi=150)
plt.close()
print("[OK] fig2_embedding_similarity.png")


print("\nDone! Figures saved to outputs/")
