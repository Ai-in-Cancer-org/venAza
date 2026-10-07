# venAza

Code accompanying the manuscript

> **Multimodal Artificial Intelligence Fusion Models to Predict Response to
> Venetoclax and Azacitidine in Patients with Newly Diagnosed Acute Myeloid Leukemia**

The pipeline predicts best response (CR/CRi vs. non-response) to frontline
venetoclax + azacitidine from bone marrow smear (BMS) images, alone and fused
with clinical and genetic variables. Two independent image branches are
implemented and each is combined with the clinical/genetic data in a late
multimodal fusion step.

No data, trained weights or results are included in this repository.

---

## Installation

```bash
git clone <this repository> && cd venAza
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
pip install git+https://github.com/Mahmoodlab/CONCH.git
huggingface-cli login     # CONCH and UNI weights are gated: request access on the
                          # Hugging Face pages of MahmoodLab/CONCH and MahmoodLab/UNI
```

Tested with Python 3.10-3.12 and PyTorch 2.x. GPU strongly recommended for
steps 3, 4, 5 and 7.

## Data

See [`data/README.md`](data/README.md) for the expected folder layout and the
clinical table (a header-only template is in `templates/`).

## Configuration

All settings live in [`configs/default.yaml`](configs/default.yaml). Create your
own config containing only the values you want to change (typically `paths`
and the `clinical.columns` mapping) and pass it to every script:

```bash
python scripts/03_train_resnet_mil.py --config configs/my_run.yaml
python scripts/03_train_resnet_mil.py --config configs/my_run.yaml --set resnet_mil.epochs=30
```

The resolved configuration is stored next to the outputs of each step.

## Running the pipeline

`bash run_pipeline.sh configs/my_run.yaml` runs everything in order
(`EXTERNAL=1` adds the external validation, `NGPU=4` runs DINO with
`torchrun`). The steps individually:

| step | script | what it does |
|---|---|---|
| 1 | `01_extract_patches.py` | Crops each ROI to a multiple of the patch size (358 px, ≤50 px discarded per border), tiles it into non-overlapping patches and applies **region-adaptive QC**: per-ROI hue/saturation/luminance statistics define the thresholds, patches dominated by cell-free background or erythrocytes are rejected. Writes kept/rejected patches and a K/R overview figure per ROI. |
| 2 | `02_prepare_cohort.py` | Maps best response to CR/CRi vs. non-response, encodes the 16 clinical/genetic features, writes the patient-level stratified 5-fold splits (one per seed, inner validation split inside the training folds) and a descriptive cohort table. |
| 3 | `03_train_resnet_mil.py` | **Custom CNN branch.** ImageNet ResNet-34 tile encoder → gated attention (tanh × sigmoid branches → scalar score per tile, softmax over all patches of the patient) → attention-weighted mean → fully connected classifier → sigmoid. Trained end to end with patient-level BCE, Adam, LR scheduling on validation AUROC, gradient clipping and mixed precision. |
| 4 | `04_dino_adapt.py` | **Self-supervised domain adaptation** (DINO, no labels) of the last two transformer blocks of CONCH ViT-B/16 (blocks 10–11) and UNI ViT-L (blocks 22–23); all other weights frozen. |
| 5 | `05_extract_features.py` | Tile embeddings with the adapted encoders. |
| 6 | `06_concat_features.py` | Concatenates CONCH and UNI embeddings (512 + 1024 = 1536 dimensions). |
| 7 | `07_train_fm_mil.py` | **Foundation-model branch.** Gated-attention MIL head on the combined embeddings (image only). |
| 8 | `08_late_fusion.py` | **Late multimodal fusion.** Second-stage classifiers (L2 logistic regression, elastic net, random forest, XGBoost) on `[image probability] + 16 features`, trained on the same outer folds using the out-of-fold image probabilities, plus the clinical/genetic-only baseline. Fold heads are stored for external application. |
| 9 | `09_evaluate.py` | AUROC, AP, accuracy, F1, sensitivity, specificity, PPV, NPV per fold (fold-specific Youden threshold) and pooled out-of-fold (pooled Youden threshold, 2000-sample bootstrap CIs); confusion matrices; ROC and PR curves (folds in colour, pooled curve in bold black); mean ± SD over seeds. |
| 10 | `10_shap.py` | Linear SHAP (correlation-dependent perturbation) per fold on the held-out patients of the LR fusion head; beeswarm and mean SHAP for true positives / true negatives (log-odds). |
| 11 | `11_statistics.py` | Univariable and multivariable logistic regression (odds ratios for CR/CRi, ELN2024 with intermediate as reference); TP/TN/FP/FN comparison (Fisher–Freeman–Halton, Kruskal–Wallis); concordance of the two fusion models (Pearson r per class, scatter plot, discordant cases). |
| 12 | `12_attention_rollout.py` | Attention rollout of the adapted ViT encoders on 224 × 224 tiles (196 patch tokens + class token) for the highest-attention tiles and as ROI reconstructions (QC-rejected tiles in grey). |
| 13 | `13_external_validation.py` | Applies all components **frozen** (image models of all seeds × folds, stored fusion heads, internal Youden thresholds) to the external cohort; no retraining, refitting or recalibration. |

### Repetitions and aggregation

All experiments are repeated for each seed in `splits.seeds`; a seed controls
weight initialisation, sampling/augmentation and fold shuffling. Step 9 reports
per-seed results (mean ± SD) and seed-aggregated per-patient predictions
(probabilities averaged across seeds), which are used by steps 10–12.

### Leakage control

* One split table per seed is shared by **all** models; image models select
  checkpoints on an inner validation split and never see their test fold.
* Fusion heads are trained on out-of-fold image probabilities of the training
  folds only and evaluated on the held-out fold.
* DINO adaptation is self-supervised and uses no response labels.

## Smoke test (no patient data, CPU, ~15 min)

```bash
python scripts/00_make_synthetic_data.py --out demo_data
EXTERNAL=1 bash run_pipeline.sh configs/smoke_test.yaml
```

`configs/smoke_test.yaml` replaces CONCH/UNI by small randomly initialised
ViTs and shortens all trainings; it only checks that every step runs. Results
on synthetic data are meaningless by construction.

## Output layout

Described at the top of [`src/pipeline.py`](src/pipeline.py); everything
is written below `paths.work_dir`.

## Repository structure

```
configs/            default.yaml (all settings), smoke_test.yaml
scripts/            numbered pipeline steps 00-13
src/
  preprocessing/    patch extraction + adaptive QC
  data/             clinical encoding, CV splits, bag datasets
  models/           gated-attention MIL models, CONCH/UNI loaders, DINO
  fusion/           late-fusion heads and cross-validated fusion
  evaluation/       metrics, confidence intervals, figures
  explainability/   SHAP, attention rollout
  analysis/         odds ratios, group comparisons
data/README.md      expected input format
templates/          clinical table template
```

## Citation

If you use this code, please cite the manuscript (reference to be added upon
publication).

## License

See `LICENSE`.
