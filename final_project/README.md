# Defending Against Generative Embedding Inversion Attacks via Gaussian Noise Perturbation

## 1. Abstract

This project extends the **Generative Embedding Inversion Attack (GEIA)** framework proposed by Li et al. (ACL 2023 Findings) by investigating **Gaussian noise perturbation** as a defense mechanism against sentence embedding inversion. We systematically evaluate how adding isotropic Gaussian noise of varying magnitudes to sentence embeddings affects the ability of a generative attacker model (DialoGPT-large) to recover the original text from the perturbed embeddings.

Our experiments across 15 noise scales ( $\sigma \in [0, 0.5]$ ) reveal three distinct regimes: **immune** ( $\sigma \in [0, 0.01]$ ), where attack quality is virtually unaffected; **transition** ( $\sigma \in (0.01, 0.05)$ ), where all metrics degrade rapidly; and **failure** ( $\sigma \in [0.05, \infty)$ ), where the attacker effectively produces random text. These findings demonstrate that while small Gaussian noise is insufficient as a practical defense, moderate noise levels ( $\sigma \geq 0.05$ ) can completely neutralize generative embedding inversion attacks — albeit at the cost of degraded embedding utility.

---

## 2. Background

### 2.1 Generative Embedding Inversion Attack (GEIA)

Sentence embeddings produced by pre-trained encoder models (e.g., Sentence-BERT) are widely used in privacy-sensitive applications such as retrieval, clustering, and feature extraction. The GEIA framework (Li et al., 2023) showed that these embeddings leak far more information than previously assumed: a **generative attacker** trained on embeddings can reconstruct the **entire original sentence**, not just high-level semantic attributes.

The attack pipeline consists of:

1. A **victim encoder** (e.g., `all-mpnet-base-v1`) that produces a fixed-dimensional sentence embedding.
2. A **projection layer** that maps the embedding into the token embedding space of a language model.
3. An **attacker decoder** (e.g., DialoGPT-large) that autoregressively generates the recovered sentence from the projected embedding.

### 2.2 Motivation: Noise as a Defense

We hypothesize that adding controlled Gaussian noise to sentence embeddings before releasing them can obscure the fine-grained information required for successful reconstruction, while ideally preserving enough semantic signal for legitimate downstream tasks. This work systematically measures the trade-off between noise magnitude and attack resilience.

---

## 3. Experimental Setup

### 3.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | PersonaChat (train + dev + test) |
| **Attacker model** | `microsoft/DialoGPT-large` (762M parameters) |
| **Victim embedding model** | `all-mpnet-base-v1` (110M parameters, 768-dim embeddings) |
| **Projection** | Linear layer: 768 → 1280 (matches GPT-2 token embedding dim) |
| **Decoding algorithm** | Beam search (beam size = 5) |
| **Training epochs** | 5 |
| **Batch size** | 16 |
| **Optimizer** | AdamW |
| **Training samples** | Full PersonaChat training set |
| **Test samples** | 1,000 (first 1,000 from test set) |
| **Noise scales ( $\sigma$ )** | 0, 0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.25, 0.5 |
| **Hardware** | NVIDIA RTX 4090 (24 GB VRAM) |
| **Training time** | ~5 hours |
| **Total experiment time** | ~7 hours (training + testing all noise levels) |
| **Python version** | 3.9 |

### 3.2 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **ROUGE-1** | Unigram-based recall-oriented F1 score |
| **ROUGE-2** | Bigram-based recall-oriented F1 score |
| **ROUGE-L** | Longest common subsequence-based F1 score |
| **BLEU-1** | Unigram precision |
| **BLEU-2** | Weighted 1‑gram + 2‑gram precision |
| **BLEU-4** | 4‑gram precision with brevity penalty (standard BLEU) |
| **Exact Match** | Proportion of exactly matching sentences |
| **Exact Match (no punct)** | Same after removing punctuation |
| **Edit Distance (mean)** | Mean Levenshtein distance |
| **Edit Distance (median)** | Median Levenshtein distance |
| **Embedding Similarity** | Cosine similarity via `sentence-t5-xxl` |

---

## 4. Implementation Details

### 4.1 Where Noise Is Injected

The noise is added **after the victim encoder's projection layer** and **before the attacker decoder's autoregressive generation**. This design ensures that the attacker receives a perturbed embedding as its initial hidden state, while the projection layer and decoder weights remain frozen during testing.

The core modification in [`scripts/test_attacker.py`](./scripts/test_attacker.py):

```python
# Victim encoding
embeddings = embedder.encode(batch_text, convert_to_tensor=True).to(device)
# Projection (768 → 1280)
embeddings = projection(embeddings)
# Noise injection (after projection, before decoding)
if noise_scale > 0:
    embeddings = embeddings + torch.randn_like(embeddings) * noise_scale
# Decode with beam search
pred_list, gt_list = eval_on_batch(
    batch_X=embeddings,
    batch_D=batch_text,
    model=config["model"],
    tokenizer=config["tokenizer"],
    device=device,
    config=config,
)
```

### 4.2 Training Details

The attacker is first trained on clean (noise-free) embeddings for 5 epochs on the full PersonaChat training set using the GEIA training pipeline ([`attacker.py`](./attacker.py)). Training loss and perplexity are printed during training for monitoring. After training, the attacker checkpoint is saved and reused across all noise-level experiments.

### 4.3 Test Data Subsampling

Following the original GEIA protocol, the test set is limited to the first 1,000 sentences ([`data_process.py`](./data_process.py), line 16–17) to balance statistical significance with computational cost.

---

## 5. Results

### 5.1 Quantitative Results

#### Figure 1: Generation Quality (ROUGE-L, BLEU-4, Exact Match)

![Generation Quality](outputs/fig1_reconstruction_quality.png)

Three key metrics plotted against noise scale on a logarithmic x-axis. The gray shaded region ( $\sigma \in (0.01, 0.05)$ ) marks the transition zone where reconstruction quality collapses.

- **ROUGE-L** drops from 0.57 (baseline) to 0.07 at $\sigma = 0.5$.
- **BLEU-4** falls below 0.01 for $\sigma \geq 0.25$.
- **Exact Match** reaches 0% for all $\sigma \in [0.05, \infty)$.

#### Figure 2: Edit Distance

![Edit Distance](outputs/fig2_edit_distance.png)

Mean edit distance rises steadily from ~25 (baseline) to ~55 at high noise levels, indicating progressively greater divergence between the reconstruction and the ground truth.

#### Figure 3: Embedding Similarity

![Embedding Similarity](outputs/fig3_embedding_similarity.png)

Cosine similarity between ground-truth and reconstructed sentence embeddings. This is the **least sensitive** metric, declining gracefully from 0.89 (baseline) to 0.63 at $\sigma = 0.5$, suggesting that even when the exact wording is lost, rough semantic content may be preserved.

### 5.2 Qualitative Results

#### Figure 4: Reconstruction Examples

![Reconstruction Examples](outputs/fig4_reconstruction_examples.png)

A side-by-side comparison of the same input sentence reconstructed at each noise level. At $\sigma \in [0, 0.01]$, the reconstructions are nearly identical to the baseline. At $\sigma \in [0.05, \infty)$, the outputs become semantically unrelated or degenerate.

### 5.3 Full Metrics Table

| $\sigma$ | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-1 | BLEU-2 | BLEU-4 | EM | EM (no punct) | ED (mean) | ED (median) | Emb. Sim |
|---------:|--------:|--------:|--------:|-------:|-------:|-------:|---:|--------------:|----------:|------------:|---------:|
| 0        |  0.6109 |  0.4137 |  0.5700 | 0.3998 | 0.2999 | 0.1731 | 0.039 |        0.154 |     25.36 |        25.0 |   0.8933 |
| 0.001    |  0.6115 |  0.4151 |  0.5714 | 0.4026 | 0.3029 | 0.1750 | 0.040 |        0.155 |     25.36 |        25.0 |   0.8936 |
| 0.002    |  0.6127 |  0.4159 |  0.5735 | 0.4044 | 0.3031 | 0.1751 | 0.040 |        0.158 |     25.32 |        25.0 |   0.8933 |
| 0.005    |  0.6063 |  0.4081 |  0.5663 | 0.3996 | 0.2985 | 0.1726 | 0.040 |        0.152 |     25.67 |        25.0 |   0.8915 |
| 0.008    |  0.5869 |  0.3914 |  0.5482 | 0.3896 | 0.2903 | 0.1634 | 0.035 |        0.143 |     26.45 |        27.0 |   0.8852 |
| 0.01     |  0.5797 |  0.3847 |  0.5423 | 0.3789 | 0.2801 | 0.1577 | 0.035 |        0.143 |     26.81 |        28.0 |   0.8833 |
| 0.015    |  0.5447 |  0.3440 |  0.5066 | 0.3648 | 0.2635 | 0.1465 | 0.027 |        0.117 |     28.51 |        30.0 |   0.8646 |
| 0.02     |  0.4817 |  0.2866 |  0.4445 | 0.3237 | 0.2254 | 0.1191 | 0.022 |        0.093 |     30.60 |        32.0 |   0.8417 |
| 0.03     |  0.3501 |  0.1651 |  0.3206 | 0.2507 | 0.1561 | 0.0686 | 0.005 |        0.020 |     36.51 |        37.0 |   0.7788 |
| 0.04     |  0.2657 |  0.0953 |  0.2409 | 0.2040 | 0.1111 | 0.0370 | 0.001 |        0.004 |     39.77 |        40.0 |   0.7335 |
| 0.05     |  0.2077 |  0.0637 |  0.1893 | 0.1635 | 0.0831 | 0.0239 | 0.000 |        0.003 |     42.33 |        42.0 |   0.7046 |
| 0.075    |  0.1355 |  0.0244 |  0.1213 | 0.1133 | 0.0462 | 0.0117 | 0.000 |        0.000 |     48.83 |        44.0 |   0.6559 |
| 0.1      |  0.1118 |  0.0169 |  0.1016 | 0.0951 | 0.0348 | 0.0044 | 0.000 |        0.000 |     51.37 |        44.0 |   0.6446 |
| 0.25     |  0.0801 |  0.0086 |  0.0721 | 0.0729 | 0.0225 | 0.0027 | 0.000 |        0.000 |     55.80 |        46.0 |   0.6274 |
| 0.5      |  0.0750 |  0.0061 |  0.0677 | 0.0706 | 0.0202 | $\sim$ 0  | 0.000 |        0.000 |     54.01 |        47.0 |   0.6257 |

---

## 6. Discussion

### 6.1 Three Regimes of Defense Effectiveness

1. **Immune regime ( $\sigma \in [0, 0.01]$ )**: All evaluation metrics remain statistically indistinguishable from the noise-free baseline. ROUGE-L stays ~0.54–0.57, and Exact Match hovers around 3.5–4.0%. Attackers are essentially unaffected by noise below this threshold.

2. **Transition regime ( $\sigma \in (0.01, 0.05)$ )**: A rapid, monotonic degradation occurs across all metrics. ROUGE-L falls from 0.51 to 0.24, BLEU-4 drops by nearly 75%, and Exact Match approaches zero. This is the region where the noise begins to overwhelm the fine-grained embedding features the attacker relies on.

3. **Failure regime ( $\sigma \in [0.05, \infty)$ )**: Exact Match reaches **0%**, BLEU-4 is near zero, and ROUGE-L falls below 0.19. The attacker outputs are effectively random with respect to the ground truth. The defense is fully effective in this regime.

### 6.2 Embedding Similarity as a Metric

Embedding similarity is notably the **least sensitive** metric — it only declines from 0.89 to 0.63 even at the highest noise level. This suggests that coarse semantic alignment can persist even when lexical accuracy is completely destroyed. Relying solely on embedding similarity would dramatically underestimate attack success.

### 6.3 Implications for Privacy-Preserving Embedding Release

- **Tiny noise is not a defense**: Noise levels below $\sigma \in [0, 0.01]$ provide no meaningful protection.
- **Moderate noise ( $\sigma \in (0.01, 0.05)$ )** introduces a trade-off between privacy and utility that may be acceptable in some applications.
- **High noise ( $\sigma \in [0.05, \infty)$ )** effectively eliminates inversion attack capability but would likely render embeddings unusable for many downstream tasks.
- Formal **differential privacy** mechanisms (e.g., adding calibrated noise with careful sensitivity analysis) would provide stronger guarantees than the ad-hoc Gaussian perturbation explored here.

---

## 7. How to Reproduce

### 7.1 Prerequisites

This project uses `uv` for dependency management. Install it via:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or refer to the [official documentation](https://docs.astral.sh/uv/#installation).

### 7.2 Environment Setup

```bash
cd final_project
uv sync
```

This creates a virtual environment and installs all dependencies (PyTorch 2.1.0, Transformers 4.36.2, Sentence-Transformers 2.2.2, etc.) as specified in `pyproject.toml`.

> [!Note]
> All commands below must be executed from the `final_project/` directory. The project requires Python 3.9 and a CUDA-capable GPU.

### 7.3 Pipeline

#### Step 1: Train the Attacker

```bash
bash scripts/01_train_attacker.sh
```

- **Input**: PersonaChat training set (`data/personachat/processed_persona/train.txt`)
- **Output**:
  - `models/attacker_gpt2_large_personachat_mpnet/`: Attacker model weights
  - `models/projection_gpt2_large_personachat_mpnet`: Projection layer weights
- **Duration**: ~5 hours on NVIDIA RTX 4090
- **Configurable arguments**: `--dataset`, `--data_type`, `--num_epochs`, `--batch_size`, `--embed_model`

#### Step 2: Run Noise Experiments

```bash
bash scripts/02_test_attacker.sh
```

- Iterates over all 15 noise scales and runs the attacker on the test set for each.
- **Input**: Trained attacker model (from Step 1) + PersonaChat test set
- **Output**: `models/attacker_gpt2_large_personachat_mpnet_beam_noise_${sigma}.log` (JSON with `gt` and `pred` lists)
- **Duration**: ~2 hours total (~10 min per noise level) on NVIDIA RTX 4090
- **Configurable arguments**: `--dataset`, `--embed_model`, `--decode`, `--batch_size`, `--noise_scale`

#### Step 3: Evaluate Results

```bash
bash scripts/03_eval_results.sh
```

- Reads all log files, computes ROUGE-L, BLEU-4, Exact Match, Edit Distance, and Embedding Similarity.
- **Input**: Log files from Step 2 (`models/attacker_gpt2_large_personachat_mpnet_beam*.log`)
- **Output**: `outputs/results.csv` (all metrics for each noise level)
- **Configurable arguments**: `--output`, `--log-dir`, `--batch-size`

#### Step 4: Generate Figures

```bash
bash scripts/04_plot_results.sh
```

- Runs [`scripts/quantitative_plot.py`](./scripts/quantitative_plot.py) (creates Figures 1–3) and [`scripts/qualitative_plot.py`](./scripts/qualitative_plot.py) (creates Figure 4).
- **Input**: `outputs/results.csv` (from Step 3)
- **Output**:
  - `outputs/fig1_reconstruction_quality.png`
  - `outputs/fig2_edit_distance.png`
  - `outputs/fig3_embedding_similarity.png`
  - `outputs/fig4_reconstruction_examples.png`

---

## 8. Project Structure

```text
final_project/
├── README.md                              # This report
├── pyproject.toml                         # Project metadata & dependencies (uv)
├── .python-version                        # Python 3.9
├── attacker.py                            # GEIA attacker training pipeline
├── attacker_evaluation_gpt.py             # Evaluation & decoding utilities
├── attacker_models.py                     # Attacker model architectures
├── data_process.py                        # Data loading & preprocessing
├── config.py                              # Shared configuration
├── decode_beam_search.py                  # Beam search decoder
├── eval_generation.py                     # Generation quality evaluation
├── simcse_persona.py                      # PersonaChat data processing
├── data/
│   └── personachat/
│       └── processed_persona/             # PersonaChat dataset splits
│           ├── train.txt
│           ├── dev.txt
│           └── test.txt
├── models/
│   ├── attacker_gpt2_large_personachat_mpnet/   # Trained attacker checkpoint
│   └── projection_gpt2_large_personachat_mpnet  # Trained projection layer
├── scripts/
│   ├── 01_train_attacker.sh               # Training script
│   ├── 02_test_attacker.sh                # Noise experiment script
│   ├── 03_eval_results.sh                 # Evaluation script
│   ├── 04_plot_results.sh                 # Visualization script
│   ├── test_attacker.py                   # Noise-injection test harness
│   ├── eval_results.py                    # Metric computation
│   ├── quantitative_plot.py               # Figures 1–3 (line charts)
│   └── qualitative_plot.py               # Figure 4 (comparison table)
├── outputs/
│   ├── results.csv                        # Full metric table
│   ├── fig1_reconstruction_quality.png    # ROUGE-L, BLEU-4, Exact Match
│   ├── fig2_edit_distance.png             # Edit distance plot
│   ├── fig3_embedding_similarity.png      # Embedding similarity plot
│   └── fig4_reconstruction_examples.png   # Qualitative comparison table
├── assets/
│   ├── results.csv                        # Full metric table
│   ├── fig1_reconstruction_quality.png    # ROUGE-L, BLEU-4, Exact Match
│   ├── fig2_edit_distance.png             # Edit distance plot
│   ├── fig3_embedding_similarity.png      # Embedding similarity plot
│   └── fig4_reconstruction_examples.png   # Qualitative comparison table 
└── ...
```

---

## 9. References

Li, H., Xu, M., & Song, Y. (2023). Sentence Embedding Leaks More Information than You Expect: Generative Embedding Inversion Attack to Recover the Whole Sentence. In *Findings of the Association for Computational Linguistics: ACL 2023*, pp. 14022–14040.

```bibtex
@inproceedings{li-etal-2023-sentence,
    title = "Sentence Embedding Leaks More Information than You Expect: Generative Embedding Inversion Attack to Recover the Whole Sentence",
    author = "Li, Haoran  and
      Xu, Mingshi  and
      Song, Yangqiu",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.881",
    doi = "10.18653/v1/2023.findings-acl.881",
    pages = "14022--14040",
}
```

---

## 10. Contributions Beyond the Original Repository

The following components were implemented as part of this project, extending the original GEIA repository with new infrastructure and scientific experiments.

### 10.1 Original Scripts

The entire [`scripts/`](./scripts/) directory was built from scratch. All files within it are original work:

- **[`01_train_attacker.sh`](./scripts/01_train_attacker.sh)** — Shell script to train the GEIA attacker on the PersonaChat dataset with configurable hyperparameters.
- **[`02_test_attacker.sh`](./scripts/02_test_attacker.sh)** — Shell script that runs the attacker across all 15 noise scales by invoking `test_attacker.py` in a loop with different `--noise_scale` values.
- **[`03_eval_results.sh`](./scripts/03_eval_results.sh)** — Shell script that runs `eval_results.py` to compute all metrics and then `quantitative_plot.py` and `qualitative_plot.py` to generate figures.
- **[`04_plot_results.sh`](./scripts/04_plot_results.sh)** — Shell script that generates all final figures from the results CSV.
- **[`test_attacker.py`](./scripts/test_attacker.py)** — The core scientific contribution: a modified test pipeline that injects Gaussian noise into projected sentence embeddings before beam-search decoding, enabling systematic measurement across noise scales.
- **[`eval_results.py`](./scripts/eval_results.py)** — Batch evaluation script that computes ROUGE-1/2/L, BLEU-1/2/4, Exact Match, Edit Distance, and Embedding Similarity by loading log files and using the `sentence-t5-xxl` model, writing the full results table to `outputs/results.csv`.
- **[`quantitative_plot.py`](./scripts/quantitative_plot.py)** — Publication-quality line chart generation (Figures 1–3) with log-scale x-axis, threshold region shading, and consistent styling.
- **[`qualitative_plot.py`](./scripts/qualitative_plot.py)** — Generates the qualitative comparison table (Figure 4) showing ground-truth and reconstructed sentences at each noise level.

### 10.2 Modifications to Original Files

Two files from the original GEIA repository were modified to support this project's experiments:

- **[`data_process.py`](./data_process.py)** — Added test set subsampling (first 1,000 sentences) to reduce experiment runtime while maintaining statistical significance.
- **[`eval_generation.py`](./eval_generation.py)** — Converted from hardcoded single-path evaluation to glob-based batch evaluation, enabling efficient processing of multiple noise-level log files.

### 10.3 Scientific Contribution

The central scientific contribution is the idea of adding isotropic Gaussian noise to sentence embeddings and systematically measuring the resulting threshold effect. By evaluating 15 noise scales across three distinct regimes (immune, transition, failure), this work provides the first detailed characterization of how Gaussian noise perturbs generative embedding inversion attacks and identifies the minimum noise level required to fully neutralize the attacker.

---

The content below is the original README from the GEIA repository, included for reference.

# GEIA

Code for Findings-ACL 2023 paper: Sentence Embedding Leaks More Information than You Expect: Generative Embedding Inversion Attack to Recover the Whole Sentence

### Package Dependencies

- numpy
- pytorch 1.10.2
- sentence_transformers 2.2.0
- transformers 4.xx.x
- simcse 0.4
- datasets

### Data Preparation

We upload PC data under the ```data/``` folder.
The ABCD dataset we experimented can be found in <https://drive.google.com/file/d/1oIo8P0Y8X9DTeEfOA1WUKq8Uix9a_Pte/view?usp=sharing>.
For other datasets, we use ```datasets``` package to download and store them, so you can run our code directly.

### Baseline Attackers

**You need to set up arguments psroperly before running codes**:
```python projection.py```

- --model_dir: Attacker model path from Huggingface (like 'gpt2-large' and 'microsoft/DialoGPT-xxxx') or local model checkpoints.
- --num_epochs: Training epoches.
- --batch_size: Batch_size #.
- --dataset: Name of the dataset including personachat, qnli, mnli, sst2, wmt16, multi_woz and abcd.
- --data_type: Train or test.
- --embed_model: The victim model you wish to attack. We currently support sentence-bert models and huggingface models, you may refer to our model_cards dictionary in ```projection.py``` for more information.
- --model_type: NN or RNN

By running:
```python projection.py```
You will train your own baseline model and evaluate it. If you want to just train or eval a certain model, check the last four lines of ```projection.py``` and disable the corresponding codes.

### GIEA

#### GPT-2 Attacker

**You need to set up arguments properly before running codes**:
```python attacker.py```

- --model_dir: Attacker model path from Huggingface (like 'gpt2-large' and 'microsoft/DialoGPT-xxxx') or local model checkpoints.
- --num_epochs: Training epoches.
- --batch_size: Batch_size #.
- --dataset: Name of the dataset including personachat, qnli, mnli, sst2, wmt16, multi_woz and abcd.
- --data_type: train or dev or test.
- --embed_model: The victim model you wish to attack. We currently support sentence-bert models and huggingface models, you may refer to our model_cards dictionary in ```attacker.py``` for more information.
- --decode: Decoding algorithm. We currently implement beam and sampling based decoding.

You should train the attacker on training data at first, then test your attacker on the test data to obtain test logs. Then you can evaluate attack performance on test logs by changing model_dir to your trained attcker and data_type to test.

If you want to train a randomly initialized GPT-2 attacker, after setting the arguments, run:
```python attacker_random_gpt2.py```

#### Other Attackers

Due to the fact that different decoders have different implementaions, we use separate py files for each model (the decoding implementations also differ).

If you want to try out opt as the attacker model, run:
```python attacker_opt.py```

If you want to try out t5 as the attacker model, run:
```python attacker_t5.py```

### Evaluation

**You need to make sure the test reuslt paths is set inside the 'eval_xxx.py' files.**

To obtain classification performance, run:
```python eval_classification.py```

To obtain generation performance, run:
```python eval_generation.py```

To calculate perplexity, you need to set the LM to caluate PPL, run:
```python eval_ppl.py```

### Citation

Please kindly cite the following paper if you found our method and resources helpful!

```
@inproceedings{li-etal-2023-sentence,
    title = "Sentence Embedding Leaks More Information than You Expect: Generative Embedding Inversion Attack to Recover the Whole Sentence",
    author = "Li, Haoran  and
      Xu, Mingshi  and
      Song, Yangqiu",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.881",
    doi = "10.18653/v1/2023.findings-acl.881",
    pages = "14022--14040",
}
```

### Miscellaneous

Please send any questions about the code and/or the algorithm to <hlibt@connect.ust.hk>
