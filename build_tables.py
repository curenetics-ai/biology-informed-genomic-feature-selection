"""
build_tables.py
===============
Turns the pickles written by corrected_nested_cv.py into exactly the numbers
needed for the revised manuscript and Supplementary Material:

  Table IV   / Table VI        performance, no SMOTE   (immunotherapy / recurrence)
  Table VII  Group, Sel., |C*| selected groups and their selection frequency (R3.2)
  Table VIII the comparisons per cohort involving the best BI method
  Suppl. II  Delta AUC from SMOTE
  Suppl. III / IV              performance with SMOTE
  Suppl. V   the complete family of comparisons per cohort (R3.3)
  Suppl. Fig. 1                six reliability-curve panels (R3.5)

Holm correction is applied WITHIN each cohort over the full family of
comparisons (12, or 15 once k-means has been run); Table VIII quotes the corrected p-values from that family, which is
what the manuscript now says.

Run after corrected_nested_cv.py has produced every cohort's pickles.
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from imblearn.metrics import classification_report_imbalanced

from utils import utils


# Cohort order as it appears in the manuscript
IMMUNO_COHORTS = ['kidney', 'melanoma', 'stomach']
RECUR_COHORTS = ['luad', 'lusc', 'nsclc']
ALL_COHORTS = IMMUNO_COHORTS + RECUR_COHORTS

COHORT_LABEL = {
    'kidney': 'Kidney Cancer', 'melanoma': 'Melanoma', 'stomach': 'Stomach Cancer',
    'luad': 'Lung Adenocarcinoma (LUAD)', 'lusc': 'Squamous Cell Lung Cancer (LUSC)',
    'nsclc': 'NSCLC (GSE103584)',
}

METHOD_LABEL = {
    'mrmr': 'mRMR', 'pca': 'PCA', 'lasso': 'LASSO', 'kmeans': 'k-means',
    'region_based': 'Region-based', 'chromosome_based': 'Chromosome-based',
    'pathway_based': 'Pathway-based',
}

# Row order used in Tables IV / VI
METHOD_ORDER = ['mrmr', 'pca', 'lasso', 'kmeans', 'region_based', 'chromosome_based', 'pathway_based']
BI_METHODS = ['region_based', 'chromosome_based', 'pathway_based']

# The 12 comparisons, taken straight from the project's own constant so the two
# can never drift apart, plus each biology-informed method against its k-means
# counterpart (skipped for a cohort whose results have no k-means yet).
COMPARISON_PAIRS = utils.FEATURE_SELECTION_COMPARISON_PAIRS + (
    ('chromosome_based', 'kmeans'), ('region_based', 'kmeans'), ('pathway_based', 'kmeans'))


_WARNED = set()


def _to_01(res, cohort, smote):
    """Recode labels to 0/1, keeping 1 as the positive class (predict_proba column 1).

    Safety net for pickles written before run_corrected.py checked its labels:
    DeLong and calibration_curve both assume 0/1 labels.
    """
    for method, m in res.items():
        classes = np.unique(m['oof_y_true'])
        if set(classes.tolist()) == {0, 1}:
            continue
        if len(classes) != 2 or 1 not in classes:
            raise ValueError('%s smote=%d %s: labels %s are not binary with 1 as the positive class'
                             % (cohort, smote, method, classes.tolist()))
        if (cohort, smote) not in _WARNED:
            _WARNED.add((cohort, smote))
            print('WARNING %s smote=%d: labels %s recoded to 0/1 (1 stays positive)'
                  % (cohort, smote, classes.tolist()))
        m['oof_y_true'] = (np.asarray(m['oof_y_true']) == 1).astype(int)
        m['oof_y_pred'] = (np.asarray(m['oof_y_pred']) == 1).astype(int)
    return res


def _load(results_dir, cohort, smote):
    path = os.path.join(results_dir, 'corrected_%s_smote%d.pkl' % (cohort, smote))
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return _to_01(pickle.load(f), cohort, smote)


def _ci(ci, scale=1.0, fmt='%.3f'):
    if ci is None or len(ci) < 2 or ci[0] is None or ci[1] is None:
        return ''
    try:
        return ('[' + fmt + ', ' + fmt + ']') % (scale * float(ci[0]), scale * float(ci[1]))
    except Exception:
        return ''


def _f1(m):
    """Support-weighted F1 of the pooled out-of-fold predictions, with its bootstrap
    95% CI -- the same F1 the submitted tables reported (utils.py avg_f1)."""
    y_true, y_pred = m['oof_y_true'], m['oof_y_pred']
    f1 = classification_report_imbalanced(y_true, y_pred, output_dict=True)['avg_f1']
    return f1, utils.get_cri_metrics_ci(y_true, y_pred)['f1_ci']


def _brier_skill(m):
    """(prevalence, Brier skill score). The reference is a model that predicts
    the cohort's positive rate p for everyone, whose Brier score is p(1 - p);
    a negative score means worse than that constant prediction."""
    p = float(np.mean(m['oof_y_true']))
    if m['brier_score'] is None or p in (0.0, 1.0):
        return p, None
    return p, 1.0 - m['brier_score'] / (p * (1.0 - p))


def _selected_group_size(m):
    """|C*| of the group shown in Table VII (the modal group), not the mean size
    over folds, which mixes in the other groups that won a fold."""
    sizes = [m['fold_n_features'][f] for f, g in m['fold_group'].items()
             if g == m['selected_group']]
    return sizes[0] if sizes else ''


# =========================================================================
# Tables IV / VI  (and Supplementary III / IV with SMOTE)
# =========================================================================
def performance_table(results_dir, cohorts, smote):
    rows = []
    for cohort in cohorts:
        res = _load(results_dir, cohort, smote)
        if res is None:
            continue
        for method in METHOD_ORDER:
            m = res.get(method)
            if m is None:
                continue
            f1, f1_ci = _f1(m)
            prevalence, bss = _brier_skill(m)
            rows.append({
                'Cohort': COHORT_LABEL[cohort],
                'Method': METHOD_LABEL[method],
                'Best model': m['best_model'],
                '#F': round(m['mean_n_features'], 1),
                # full precision throughout, so the manuscript tables are rounded only once
                'Acc': '%.4f' % (100 * m['acc']),
                'Acc 95% CI': _ci(m['acc_ci'], scale=100, fmt='%.4f'),
                'F1': '%.4f' % (100 * f1),
                'F1 95% CI': _ci(f1_ci, scale=100, fmt='%.4f'),
                'AUC': '%.6f' % m['AUC'],
                'AUC 95% CI': _ci(m['AUC_ci'], fmt='%.6f'),
                'PR-AUC': '%.6f' % m['PR_AUC'],
                'PR-AUC 95% CI': _ci(m['PR_AUC_ci'], fmt='%.6f'),
                'Brier': '%.6f' % m['brier_score'] if m['brier_score'] is not None else '',
                'Positives (%)': '%.1f' % (100 * prevalence),
                'Brier skill': '%.6f' % bss if bss is not None else '',
                'Cal. int.': '%.6f' % m['calib_intercept'] if m['calib_intercept'] is not None else '',
                'Cal. slope': '%.6f' % m['calib_slope'] if m['calib_slope'] is not None else '',
            })
    return pd.DataFrame(rows)


# =========================================================================
# Table VII  --  selected groups and selection frequency  (R3.2)
# =========================================================================
def selected_groups_table(results_dir, smote=0):
    rows = []
    for cohort in ALL_COHORTS:
        res = _load(results_dir, cohort, smote)
        if res is None:
            continue
        row = {'Cohort': COHORT_LABEL[cohort]}
        for method in BI_METHODS:
            m = res.get(method)
            short = {'chromosome_based': 'Chr', 'region_based': 'Region',
                     'pathway_based': 'Pathway'}[method]
            if m is None:
                row['%s Group' % short] = ''
                row['%s Sel.' % short] = ''
                row['%s |C*|' % short] = ''
            else:
                row['%s Group' % short] = m['selected_group']
                row['%s Sel.' % short] = '%d/%d' % (m['selection_count'], m['n_folds_used'])
                row['%s |C*|' % short] = _selected_group_size(m)
        rows.append(row)
    return pd.DataFrame(rows)


def group_stability_detail(results_dir, smote=0):
    """Every group that won at least one fold, per cohort and taxonomy."""
    rows = []
    for cohort in ALL_COHORTS:
        res = _load(results_dir, cohort, smote)
        if res is None:
            continue
        for method in BI_METHODS:
            m = res.get(method)
            if m is None:
                continue
            for group, count in sorted(m['group_counts'].items(), key=lambda kv: -kv[1]):
                rows.append({'Cohort': COHORT_LABEL[cohort],
                             'Taxonomy': METHOD_LABEL[method],
                             'Group': group, 'Folds won': count,
                             'of': m['n_folds_used']})
    return pd.DataFrame(rows)


# =========================================================================
# Supplementary Table V  --  the complete family of comparisons  (R3.3)
# and Table VIII, which quotes 3 of them per cohort
# =========================================================================
def paired_comparisons(results_dir, smote=0, n_bootstraps=utils.BOOTSTRAP_N_RESAMPLES):
    rows = []

    for cohort in ALL_COHORTS:
        res = _load(results_dir, cohort, smote)
        if res is None:
            continue

        cohort_rows = []
        p_by_test = {t: [] for t in utils._PAIRWISE_TEST_NAMES}
        idx_by_test = {t: [] for t in utils._PAIRWISE_TEST_NAMES}

        for type_a, type_b in COMPARISON_PAIRS:
            a, b = res.get(type_a), res.get(type_b)
            if a is None or b is None:
                continue

            # the paired tests are only valid if both sides scored the same
            # samples in the same order -- they do, because every method uses
            # the same seeded outer splits, but check rather than assume
            if not np.array_equal(a['oof_y_true'], b['oof_y_true']):
                cohort_rows.append({
                    'Cohort': COHORT_LABEL[cohort], 'Method A': METHOD_LABEL[type_a],
                    'Method B': METHOD_LABEL[type_b],
                    'note': 'skipped: outer-fold sample order differs'})
                continue

            entry = utils._compare_two_predictions(
                a['oof_y_true'], a['oof_y_pred'], a['oof_y_prob'],
                b['oof_y_pred'], b['oof_y_prob'], n_bootstraps=n_bootstraps)

            row = {
                'Cohort': COHORT_LABEL[cohort],
                'Method A': METHOD_LABEL[type_a],
                'Method B': METHOD_LABEL[type_b],
                'AUC A': round(a['AUC'], 6),
                'AUC B': round(b['AUC'], 6),
                'dAUC': round(a['AUC'] - b['AUC'], 6),
                'dPR-AUC': round(a['PR_AUC'] - b['PR_AUC'], 6),
            }
            for test_name, d in entry.items():
                row['p_%s_raw' % test_name] = d.get('p_raw', np.nan)
                if test_name == 'bootstrap_pr_auc':
                    # utils' bootstrap returns the CI of B - A (its 'model' minus its
                    # 'baseline'); negate and swap it so it matches dPR-AUC = A - B
                    row['dPR-AUC CI'] = '[%.6f, %.6f]' % (-d.get('ci_high', np.nan),
                                                          -d.get('ci_low', np.nan))
                if np.isfinite(d.get('p_raw', np.nan)):
                    p_by_test[test_name].append(d['p_raw'])
                    idx_by_test[test_name].append(len(cohort_rows))
            cohort_rows.append(row)

        # Holm WITHIN this cohort, over the whole family, per test
        for test_name in utils._PAIRWISE_TEST_NAMES:
            if not idx_by_test[test_name]:
                continue
            adj = utils.holm_correction(p_by_test[test_name])
            for i, a_val in zip(idx_by_test[test_name], adj):
                cohort_rows[i]['p_%s_holm' % test_name] = float(a_val)

        rows.extend(cohort_rows)

    return pd.DataFrame(rows)


def table_viii(full_comparisons, results_dir, smote=0):
    """The rows per cohort that compare that cohort's best BI method with each
    data-driven method.

    'Best' here means the highest AUC among the three biology-informed methods,
    which is a descriptive choice of what to display -- the p-values still come
    from the cohort's full comparison family, as the manuscript states.
    """
    keep = []
    for cohort in ALL_COHORTS:
        res = _load(results_dir, cohort, smote)
        if res is None:
            continue
        aucs = {m: res[m]['AUC'] for m in BI_METHODS if m in res}
        if not aucs:
            continue
        best_bi = METHOD_LABEL[max(aucs, key=aucs.get)]
        sub = full_comparisons[
            (full_comparisons['Cohort'] == COHORT_LABEL[cohort]) &
            (full_comparisons['Method A'] == best_bi) &
            (full_comparisons['Method B'].isin(['mRMR', 'PCA', 'LASSO', 'k-means']))]
        keep.append(sub)
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()


# =========================================================================
# Supplementary Table II  --  effect of SMOTE
# =========================================================================
def smote_delta_table(results_dir):
    rows = []
    for cohort in ALL_COHORTS:
        r0, r1 = _load(results_dir, cohort, 0), _load(results_dir, cohort, 1)
        if r0 is None or r1 is None:
            continue
        row = {'Cohort': COHORT_LABEL[cohort]}
        for method in METHOD_ORDER:
            if method in r0 and method in r1:
                row[METHOD_LABEL[method]] = round(r1[method]['AUC'] - r0[method]['AUC'], 6)
            else:
                row[METHOD_LABEL[method]] = ''
        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================
# Supplementary  --  AUC within each study of a pooled cohort
# =========================================================================
def per_study_table(results_dir, smote=0):
    """AUC within each contributing study, from the same out-of-fold predictions
    as the pooled AUC. Response rates differ between studies, so this shows
    whether the pooled result holds inside each one. Pooled cohorts only
    (pickles that carry oof_study)."""
    rows = []
    for cohort in ALL_COHORTS:
        res = _load(results_dir, cohort, smote)
        if res is None:
            continue
        for method in METHOD_ORDER:
            m = res.get(method)
            if m is None or m.get('oof_study') is None:
                continue
            study = np.asarray(m['oof_study'])
            for s in np.unique(study):
                sel = study == s
                y_true, y_prob = m['oof_y_true'][sel], m['oof_y_prob'][sel, 1]
                rows.append({
                    'Cohort': COHORT_LABEL[cohort],
                    'Method': METHOD_LABEL[method],
                    'Study': s,
                    'n': int(sel.sum()),
                    'Responders': int(y_true.sum()),
                    'AUC': (round(roc_auc_score(y_true, y_prob), 6)
                            if len(np.unique(y_true)) == 2 else np.nan),
                    'Pooled AUC': round(m['AUC'], 6),
                })
    return pd.DataFrame(rows)


# =========================================================================
# Supplementary Figure 1  --  six reliability curves  (R3.5)
# =========================================================================
CURVE_COLOR, INK, MUTED = '#2a78d6', '#222222', '#6b6b6b'


def _wilson(k, n, z=1.96):
    """95% Wilson interval for k positives out of n."""
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def _calibration_bins(y_true, y_prob, n_bins):
    """Equal-count bins of the predicted probability. One row per bin: mean
    prediction, observed fraction of positives, its 95% Wilson CI, n."""
    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:                                   # every prediction identical
        edges = np.array([y_prob.min(), y_prob.min() + 1e-12])
    idx = np.clip(np.searchsorted(edges, y_prob, side='right') - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            n, k = int(sel.sum()), int(y_true[sel].sum())
            lo, hi = _wilson(k, n)
            rows.append((y_prob[sel].mean(), k / n, lo, hi, n))
    return np.array(rows)


def reliability_figure(results_dir, out_path, smote=0, method='pathway_based', n_bins=5):
    """One panel per cohort: the observed fraction of positives against the mean
    predicted probability in n_bins equal-count bins, with 95% Wilson CIs, the
    diagonal (perfect calibration) and the cohort's positive rate (what a model
    with no information would predict). A rug shows every out-of-fold
    prediction, positives along the top and negatives along the bottom, so the
    overlap between the two is visible. Each title gives n, the positive rate,
    the Brier score, the Brier skill score and the calibration slope."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), sharex=True, sharey=True)
    for ax, cohort in zip(axes.ravel(), ALL_COHORTS):
        res = _load(results_dir, cohort, smote)
        ax.plot([0, 1], [0, 1], ls='--', lw=1, color=MUTED)
        title = COHORT_LABEL[cohort]
        if res is not None and method in res:
            m = res[method]
            y_true = np.asarray(m['oof_y_true'])
            y_prob = np.asarray(m['oof_y_prob'])[:, 1]
            p, bss = _brier_skill(m)
            ax.axhline(p, ls=':', lw=1.3, color=MUTED)
            for outcome, y_rug in ((1, 1.03), (0, -0.03)):          # positives on top
                probs = y_prob[y_true == outcome]
                ax.plot(probs, np.full_like(probs, y_rug), '|', ms=7, mew=0.7,
                        color=MUTED, alpha=0.55, clip_on=False)
            b = _calibration_bins(y_true, y_prob, n_bins)
            ax.errorbar(b[:, 0], b[:, 1], yerr=[b[:, 1] - b[:, 2], b[:, 3] - b[:, 1]],
                        fmt='o-', color=CURVE_COLOR, lw=2, ms=7, mec='white', mew=1.2,
                        elinewidth=1, capsize=3, solid_capstyle='round', zorder=3)
            title += ('\nn = %d, %.0f%% positive | Brier %.3f, skill %s, slope %.2f'
                      % (len(y_true), 100 * p, m['brier_score'],
                         '%.2f' % bss if bss is not None else 'n/a', m['calib_slope']))
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.06, 1.06)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(alpha=0.25, lw=0.5)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8.5, colors=INK)
    for ax in axes[-1]:
        ax.set_xlabel('Mean predicted probability (%d equal-count bins)' % n_bins, fontsize=9.5, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel('Observed fraction of positives', fontsize=9.5, color=INK)

    handles = [
        plt.Line2D([], [], color=CURVE_COLOR, marker='o', lw=2, ms=7, mec='white', mew=1.2,
                   label='%s: observed rate per bin, 95%% CI' % METHOD_LABEL[method]),
        plt.Line2D([], [], color=MUTED, ls='--', lw=1, label='Perfect calibration'),
        plt.Line2D([], [], color=MUTED, ls=':', lw=1.3, label='Positive rate (no-information prediction)'),
        plt.Line2D([], [], color=MUTED, marker='|', ls='', ms=7,
                   label='Predictions: positives (top), negatives (bottom)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('saved %s' % out_path)


# =========================================================================
# Write everything to one workbook
# =========================================================================
def build_all(results_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    xls_path = os.path.join(out_dir, 'corrected_results_for_manuscript.xlsx')

    full0 = paired_comparisons(results_dir, smote=0)

    sheets = {
        'TableIV_immuno_noSMOTE': performance_table(results_dir, IMMUNO_COHORTS, 0),
        'TableVI_recur_noSMOTE': performance_table(results_dir, RECUR_COHORTS, 0),
        'TableVII_selected_groups': selected_groups_table(results_dir, 0),
        'TableVII_detail_stability': group_stability_detail(results_dir, 0),
        'TableVIII_paired_best_BI': table_viii(full0, results_dir, 0),
        'SupplII_SMOTE_deltaAUC': smote_delta_table(results_dir),
        'SupplIII_immuno_SMOTE': performance_table(results_dir, IMMUNO_COHORTS, 1),
        'SupplIV_recur_SMOTE': performance_table(results_dir, RECUR_COHORTS, 1),
        'SupplV_full_12_family': full0,
        'Suppl_per_study_AUC': per_study_table(results_dir, 0),
    }

    with pd.ExcelWriter(xls_path, engine='openpyxl') as writer:
        for name, df in sheets.items():
            (df if not df.empty else pd.DataFrame({'note': ['no results']})
             ).to_excel(writer, sheet_name=name[:31], index=False)
    print('saved %s' % xls_path)

    reliability_figure(results_dir, os.path.join(out_dir, 'calibration_six_panels.png'), smote=0)
    return xls_path


if __name__ == '__main__':
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'corrected_results'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'manuscript_tables'
    build_all(results_dir, out_dir)
