# Experiments to run, and the code that runs them

Three files go next to your existing scripts (where `from utils import utils`
already resolves):

| File | What it is |
|---|---|
| `corrected_nested_cv.py` | The corrected nested CV engine (R3.1) |
| `build_tables.py` | Turns the results into the manuscript tables and the calibration figure |
| `run_corrected.py` | Driver: loads each cohort exactly as your scripts do |

```
python run_corrected.py kidney      # one cohort at a time
python run_corrected.py melanoma
python run_corrected.py stomach
python run_corrected.py luad
python run_corrected.py lusc
python run_corrected.py nsclc
python run_corrected.py tables      # after all six finish
```

Edit `BASE` at the top of `run_corrected.py` first. Cohorts are independent, so
they can run on different machines and the pickles be copied into one
`corrected_results/` folder before the `tables` step.

---

## There is one experiment, not several

No new cohort, no new method, no new data. The same 6 cohorts, 3 biology-informed
taxonomies, 3 data-driven baselines and 9 classifiers, re-run under the corrected
protocol, twice (SMOTE off and on). Everything the revision needs comes out of
that one re-run.

### What the corrected protocol changes

Inside **every outer training fold**, using only that fold's training data:

1. **Φ (Eq. 3)** scores every group of the taxonomy by the mean balanced accuracy
   of a logistic-regression model over the 5 inner folds; **Eq. (4)** keeps the
   top group. The retained group may now differ from fold to fold.
2. The **classifier and its hyperparameters** are chosen by the same inner CV.
3. For **mRMR / LASSO / PCA**, the number of features or components is chosen by
   the same inner CV.

The outer test fold is touched once, to produce the predictions that are scored.

> The group *definitions* (which gene is on which chromosome, band or pathway)
> come from external databases and carry no label information, so they are still
> built once outside the loop. What moved inside the fold is the *choice* among
> them. Worth saying plainly if the reviewer asks.

### What it fills

| Output | Sheet produced by `build_tables.py` |
|---|---|
| Table IV (kidney, melanoma, stomach) | `TableIV_immuno_noSMOTE` |
| Table VI (LUAD, LUSC, NSCLC) | `TableVI_recur_noSMOTE` |
| Table VII — Group, **Sel.**, \|C*\| | `TableVII_selected_groups` |
| Table VIII — 3 comparisons per cohort | `TableVIII_paired_best_BI` |
| Suppl. Table II — ΔAUC from SMOTE | `SupplII_SMOTE_deltaAUC` |
| Suppl. Tables III / IV — with SMOTE | `SupplIII_immuno_SMOTE`, `SupplIV_recur_SMOTE` |
| Suppl. Table V — full 12-comparison family | `SupplV_full_12_family` |
| Suppl. Fig. 1 — six reliability panels | `calibration_six_panels.png` / `.pdf` |

A bonus sheet, `TableVII_detail_stability`, lists every group that won at least
one fold. If a reviewer asks how stable the selection is, that is the answer.

### Runtime

Φ is cheap — one logistic regression per group per inner fold — so ranking ~1,600
pathways across 10 outer folds is roughly 80,000 small fits, minutes with
`N_JOBS=-1`. The expensive part is the 9-classifier grid search, which now runs
**once per outer fold per method** instead of once per group, so the corrected
protocol is substantially *faster* than the submitted one. Budget a few hours per
cohort per SMOTE setting; SMOTE roughly doubles it.

Nothing is cached between runs, unlike `utils.py`, which reloads fitted models
from `Models/`. If you re-run after changing anything, you get a genuinely fresh
result.

### Verification already done

The pipeline was run end to end on a synthetic cohort with one deliberately
predictive gene set planted in it. Φ selected that set in every fold
(`Sel. 3/3`), the non-predictive taxonomies selected a different group in each
fold, all six methods produced out-of-fold predictions in the same sample order
(so the paired tests are valid), and the full 72-row comparison family with Holm
correction, all nine sheets and the six-panel figure were generated.

---

## Four things in the code that need your decision

These are not reviewer comments. They surfaced while reading `utils.py` against
the manuscript, and three of them will be visible to the reviewer the moment the
repository is public (R3.4).

### 1. Φ is not in the supplied code

`utils.py` has no group-ranking function. It runs all 9 classifiers over every
group and then calls `select_best_per_fs_type`, which takes the **argmax of
pooled out-of-fold AUC** — the group, the classifier and the feature count all
chosen on the predictions they are scored on.

But the groups reported in the paper are not that argmax either: the kidney
pathway ranks 470th of 1,611 by outer-fold AUC. The reported groups are the ones
hard-coded in the commented-out blocks (`REACTOME_LIPOPHAGY`, chromosome 8, band
p13.31 for kidney; `REACTOME_DOWNREGULATION_OF_TGF_BETA_RECEPTOR_SIGNALING` for
LUAD; `REACTOME_REGULATION_OF_MECP2_EXPRESSION_AND_ACTIVITY` for LUSC).

**So where did those groups come from?** If Φ exists in a script not in this
archive, it should go in the repository. If the groups were chosen another way,
Section III describes a method that was not the one used, and that is a bigger
problem than R3.1 as the reviewer framed it. `corrected_nested_cv.py` implements
Φ exactly as Section III specifies, so from this run on, code and manuscript
agree.

### 2. Table II does not match the code — in every row

| Table II says | `get_classifiers()` actually uses |
|---|---|
| SVM: C∈{0.1,1,10}, kernel{linear,RBF}, γ{scale,auto} | two separate entries: linear C∈{0.01,0.1,1,10}; RBF C∈{0.1,1,10}, γ∈{0.001,0.01,0.1,0.5,1,2,5} |
| LR: C∈{0.01,0.1,1,10}, liblinear | C∈{0.1,1,10}, lbfgs, `class_weight='balanced'` |
| kNN: k∈{3,5,7,9}, weights{uniform,distance} | k∈{3,5,7,9} only |
| RF / ET: trees{200,500}, depth{None,5,10} | RF trees{50,100,200}; ET trees{50,100,200}, depth{None,10,20} |
| GB / XGB: lr{0.01,0.1}, depth{2,3,5}, trees{100,300} | GB trees{50,100,200}, lr{0.01,0.1,0.2} |
| AB: estimators{50,100,200}, lr{0.5,1.0} | estimators{100,200,500}, no lr grid |
| LDA: solver{svd,lsqr}, shrinkage{None,auto} | no grid at all |

**XGBoost is named in the manuscript (Section IV and `\cite{chen2016xgboost}`)
but does not exist in the code.** The nine are SVM-linear, SVM-RBF, RF, LR, kNN,
GB, AdaBoost, ExtraTrees, LDA.

Two ways out, and it is your call:
- **Add XGBoost** to `get_classifiers()` and align the grids to Table II, then
  re-run. The manuscript stays as it is.
- **Correct Table II and the classifier sentence** to describe what the code
  does. Smaller change, but it touches text the reviewers did not query.

Either way this must be settled before the repository goes public — a reviewer
who opens it will see Table II does not describe the code. I have left the
manuscript untouched, since guessing which way you want it would be worse than
asking.

### 3. PCA retains 96%, not 95%

The manuscript says 95% of variance. The driver passes `0.95`, but
`_precompute_single_fold` ignores it and hard-codes `PCA(n_components=0.96)`.
`corrected_nested_cv.py` uses the grid value, so it honours 0.95; change
`PCA_VARIANCE_GRID` if you prefer 0.96.

### 4. mRMR is effectively an F-test ranking

`utils.py` calls `mrmr_classif(..., K=5, return_scores=True)` and then keeps the
top `feat_name` genes by the **relevance score**. For any feature count above 5,
the redundancy term never applies, so the baseline labelled mRMR is a univariate
relevance ranking. The behaviour is reproduced in the new code so the numbers
stay comparable; set `MRMR_K_FULL = True` to run mRMR properly — but then the
manuscript needs a sentence saying so.

---

## Still outstanding, unrelated to the code

- **R3.4** — the repository URL is still `REPOSITORY-TO-BE-INSERTED` in the
  manuscript, and the note to yourselves is still in the R3.4 answer of the
  response letter. Both must go before submission.
- **R3.7** — the seven flagged references are fixed and verified, plus one
  further error found independently. The remaining entries still need an
  entry-by-entry check; 39 of the 88 use `and others`.
- Once the numbers are in, the 13 `Space reserved for` markers in the manuscript
  need writing, following the rule promised under R3.6: numerical differences
  described as numerical, and no absence of significance presented as evidence of
  equivalence.
