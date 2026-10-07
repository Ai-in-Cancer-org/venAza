#!/usr/bin/env bash
# =============================================================================
# Full pipeline. Usage:  bash run_pipeline.sh configs/my_run.yaml
# Each step can also be run on its own (see README). GPU-heavy steps
# (3, 4, 5, 7) are typically submitted as separate cluster jobs.
# =============================================================================
set -euo pipefail
CFG=${1:-configs/default.yaml}
PY=${PYTHON:-python}
NGPU=${NGPU:-1}

# --- development cohort ------------------------------------------------------
$PY scripts/01_extract_patches.py   --config "$CFG"           # patches + adaptive QC
$PY scripts/02_prepare_cohort.py    --config "$CFG"           # labels, features, splits

$PY scripts/03_train_resnet_mil.py  --config "$CFG"           # CNN branch (all seeds x folds)

for ENC in conch uni; do                                      # FM branch
  if [ "$NGPU" -gt 1 ]; then
    torchrun --nproc_per_node="$NGPU" scripts/04_dino_adapt.py --config "$CFG" --encoder $ENC
  else
    $PY scripts/04_dino_adapt.py --config "$CFG" --encoder $ENC
  fi
  $PY scripts/05_extract_features.py --config "$CFG" --encoder $ENC
done
$PY scripts/06_concat_features.py   --config "$CFG"
$PY scripts/07_train_fm_mil.py      --config "$CFG"

$PY scripts/08_late_fusion.py       --config "$CFG"           # clinical-only + fusion heads
$PY scripts/09_evaluate.py          --config "$CFG"           # tables, ROC / PR curves
$PY scripts/10_shap.py              --config "$CFG"           # SHAP (LR fusion)
$PY scripts/11_statistics.py        --config "$CFG"           # odds ratios, error groups, concordance
$PY scripts/12_attention_rollout.py --config "$CFG"           # rollout figures

# --- external validation (optional) ------------------------------------------
if [ "${EXTERNAL:-0}" = "1" ]; then
  $PY scripts/01_extract_patches.py  --config "$CFG" --cohort external
  for ENC in conch uni; do
    $PY scripts/05_extract_features.py --config "$CFG" --encoder $ENC --cohort external
  done
  $PY scripts/06_concat_features.py  --config "$CFG" --cohort external
  $PY scripts/13_external_validation.py --config "$CFG"
fi
