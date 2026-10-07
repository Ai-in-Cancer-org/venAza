# Input data

Patient data are **not** part of this repository. Image data are available
upon reasonable request from the corresponding author (see the manuscript).

## Expected layout

```
data/
├── rois/                         # development cohort
│   ├── <patient_id>/
│   │   ├── <roi_1>.png           # manually selected regions of interest
│   │   └── <roi_2>.png           # (png / jpg / tif), exported at full resolution
│   └── ...
├── clinical.csv                  # one row per patient (see below)
└── external/                     # optional external validation cohort
    ├── rois/<patient_id>/*.png
    └── clinical.csv
```

* Folder names under `rois/` must match the `patient_id` column.
* ROIs should be exported with width/height close to a multiple of the patch
  size (default 358 px); at most `max_border_crop` pixels per dimension are
  discarded when cropping to the patch grid, otherwise the ROI is skipped.

## Clinical table

A header-only template is in `templates/clinical_template.csv`. Column names
can differ - map them in the `clinical.columns` / `clinical.mutations` blocks
of the configuration.

| column (default name) | content |
|---|---|
| `patient_id` | identifier, identical to the ROI folder name |
| `best_response` | best response; CR / CRi = responder, PR / refractory / ... = non-responder (lists in the config) |
| `eln2024_li` | ELN 2024 less-intensive risk: favorable / intermediate / adverse |
| `swog_cyto` | SWOG cytogenetic risk: favorable / intermediate / unfavorable / unknown |
| `age_at_diagnosis` | years |
| `bm_blasts` | bone marrow blasts, percent (0-100) or fraction (0-1); ranges such as `30-40%` use the midpoint |
| `sex` | optional (M / F), descriptive statistics only |
| `NPM1`, `IDH1_2`, `TP53`, `ASXL1`, `RAS`, `PTPN11`, `RUNX1`, `FLT3_ITD` | Y / N (or 1 / 0) |

Categories that cannot be assigned (e.g. SWOG *unknown*) are encoded as the
all-zero indicator vector (reference level) - they are neither dropped nor imputed.
Rows whose response cannot be mapped are excluded and reported.
