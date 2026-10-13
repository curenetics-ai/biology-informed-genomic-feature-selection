import os
from tqdm.auto import tqdm

from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, GridSearchCV, ParameterGrid
from sklearn.preprocessing import label_binarize

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score, \
    brier_score_loss
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import pandas as pd
import numpy as np

import pickle

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from mrmr import mrmr_classif

from scipy.stats import norm, chi2, binom, rankdata

# Number of paired resamples for the PR-AUC bootstrap test (requested: 10,000)
BOOTSTRAP_N_RESAMPLES = 10000

# Number of parallel worker processes used for fold pre-computation and model fitting.
#   -1 -> all CPU cores,  -2 -> all but one,  1 -> sequential (original behaviour)
N_JOBS = -1

# Bootstrap resamples are evaluated in vectorised blocks of this many rows
# (memory per block ~ BOOTSTRAP_CHUNK_SIZE * n_samples * 8 bytes per temporary array).
BOOTSTRAP_CHUNK_SIZE = 1000

# Which classifier(s) act as the baseline in the paired tests (Sheet 3: Classifier_Pairwise_Tests).
#   'all'                              -> every classifier is a baseline (all pairwise comparisons)
#   ['RandomForest', 'SVM_rbf', ...]   -> only these are used as baselines
#   'RandomForest'                     -> a single baseline (original behaviour)
# Symmetric pairs are computed once and then shown from each baseline's point of view,
# so 'all' costs C(n_models, 2) comparisons, not n_models * (n_models - 1).
BASELINE_MODELS = 'all'

# Feature-selection METHODS compared against each other in Sheet 5 (FeatureSelection_Comparison).
# config_name (the key of results_all) is built by the driver scripts as
# `fs_method + str(feat_name)` with NO separator (e.g. results_all[fs_method+str(feat_name)]
# in luad bi.py / lusc traditionnal_ml.py), so config_name is expected to start
# with the exact fs_method string, e.g. "chromosome_basedchr num 7", "region_basedchr1_100",
# "pathway_basedKEGG_x", "mrmr50", "pca0.95", "lasso20".
FEATURE_SELECTION_TYPES = ('chromosome_based', 'region_based', 'pathway_based', 'mrmr', 'pca', 'lasso')

# The 12 feature-selection-type comparisons to run in Sheet 5, each side represented by its
# own best-performing classifier (see FS_TYPE_SELECTION_METRIC below).
FEATURE_SELECTION_COMPARISON_PAIRS = (
    ('chromosome_based', 'pca'),
    ('region_based', 'pca'),
    ('pathway_based', 'pca'),
    ('chromosome_based', 'lasso'),
    ('region_based', 'lasso'),
    ('pathway_based', 'lasso'),
    ('chromosome_based', 'mrmr'),
    ('region_based', 'mrmr'),
    ('pathway_based', 'mrmr'),
    ('chromosome_based', 'region_based'),
    ('pathway_based', 'region_based'),
    ('pathway_based', 'chromosome_based'),
)

# Metric used to pick each feature-selection type's "own best classifier" for Sheet 5.
# Must be a key present in evaluate_single_model_kfold's pooled results_dict (e.g. 'AUC', 'PR_AUC', 'f1').
FS_TYPE_SELECTION_METRIC = 'AUC'

# Excel allows 1,048,576 rows per sheet (incl. the header row); larger sheets are written to CSV.
EXCEL_MAX_DATA_ROWS = 1048576 - 1

# Tests run in both Sheet 3 (classifiers within a config) and Sheet 5 (feature-selection types).
_PAIRWISE_TEST_NAMES = ('delong', 'mcnemar', 'bootstrap_pr_auc')


# -------------------------------------------------------------------------
# Vectorised bootstrap kernels
#
# Every bootstrap below used to call an sklearn/imblearn metric once per resample
# (1,000 - 10,000 Python-level calls, each with heavy input validation). The kernels
# here evaluate a whole block of resamples at once with NumPy and give the same
# numbers. Resample indices are drawn from the same RandomState stream as before
# (rng.randint(0, n, (b, n)) yields exactly the values of b successive
# rng.randint(0, n, n) calls), so CIs / p-values are reproduced exactly.
# -------------------------------------------------------------------------
def _iter_bootstrap_blocks(rng, n, n_bootstraps, chunk_size=BOOTSTRAP_CHUNK_SIZE):
    done = 0
    while done < n_bootstraps:
        b = min(chunk_size, n_bootstraps - done)
        yield rng.randint(0, n, (b, n))
        done += b


def _has_two_classes(y_rows):
    """Row-wise equivalent of `len(np.unique(row)) >= 2`."""
    return ~(y_rows == y_rows[:, :1]).all(axis=1)


def _is_binary_01(y_true):
    return set(np.unique(y_true).tolist()) <= {0, 1}


def _auc_rows(pos, scores):
    """Row-wise ROC-AUC (Mann-Whitney U with mid-ranks for ties, identical to roc_auc_score).
    pos: (B, n) bool positive-class mask, scores: (B, n) float."""
    n = pos.shape[1]
    n1 = pos.sum(axis=1)
    n0 = n - n1
    ranks = rankdata(scores, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return ((ranks * pos).sum(axis=1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _average_precision_rows(pos, scores):
    """Row-wise average precision, identical to sklearn's average_precision_score:
    AP = sum_k (R_k - R_{k-1}) * P_k over the distinct score thresholds."""
    n = pos.shape[1]
    order = np.argsort(-scores, axis=1, kind='mergesort')
    s = np.take_along_axis(scores, order, axis=1)
    lab = np.take_along_axis(pos, order, axis=1)
    tp = np.cumsum(lab, axis=1, dtype=float)
    precision = tp / np.arange(1, n + 1)
    # last position of every block of tied scores = one threshold
    is_threshold = np.ones_like(lab, dtype=bool)
    is_threshold[:, :-1] = s[:, :-1] != s[:, 1:]
    # tp at the previous threshold (tp is non-decreasing, so a running max is a forward fill)
    tp_at_thr = np.maximum.accumulate(np.where(is_threshold, tp, 0.0), axis=1)
    prev_tp = np.zeros_like(tp)
    prev_tp[:, 1:] = tp_at_thr[:, :-1]
    n1 = tp[:, -1]
    with np.errstate(divide='ignore', invalid='ignore'):
        return (is_threshold * precision * (tp - prev_tp)).sum(axis=1) / n1


def _balanced_accuracy_rows(yt, yp, labels):
    """Row-wise balanced accuracy (mean recall over the classes present in y_true),
    identical to balanced_accuracy_score."""
    recall_sum = np.zeros(yt.shape[0])
    n_present = np.zeros(yt.shape[0])
    for c in labels:
        t = yt == c
        support = t.sum(axis=1)
        tp = (t & (yp == c)).sum(axis=1)
        present = support > 0
        recall_sum += np.where(present, tp / np.maximum(support, 1), 0.0)
        n_present += present
    return recall_sum / n_present


def _cri_avg_rows(yt, yp, labels):
    """Row-wise support-weighted avg f1 / precision / recall / specificity, identical to
    classification_report_imbalanced(...)['avg_f1'|'avg_pre'|'avg_rec'|'avg_spe']
    (zero-division -> 0, classes absent from y_true get zero weight)."""
    n = yt.shape[1]
    sums = np.zeros((4, yt.shape[0]))
    total_support = np.zeros(yt.shape[0])
    with np.errstate(divide='ignore', invalid='ignore'):
        for c in labels:
            t = yt == c
            p = yp == c
            tp = (t & p).sum(axis=1).astype(float)
            support = t.sum(axis=1).astype(float)
            pred_count = p.sum(axis=1).astype(float)
            fp = pred_count - tp
            tn = n - support - fp
            pre = np.where(pred_count > 0, tp / pred_count, 0.0)
            rec = np.where(support > 0, tp / support, 0.0)
            f1 = np.where(pre + rec > 0, 2 * pre * rec / (pre + rec), 0.0)
            spe = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
            sums += np.vstack((f1, pre, rec, spe)) * support
            total_support += support
    return sums / total_support


# -------------------------------------------------------------------------
# Helper Functions for 95% Confidence Intervals via Bootstrapping
# -------------------------------------------------------------------------
def _percentile_ci(scores, alpha):
    sorted_scores = np.sort(scores)
    return (np.percentile(sorted_scores, (alpha / 2) * 100),
            np.percentile(sorted_scores, (1 - alpha / 2) * 100))


def get_bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, alpha=0.05, n_jobs=-1, **kwargs):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(42)

    # Fast vectorised paths for the metrics used in this module (same resamples, same result)
    fast = None
    if not kwargs:
        if metric_func is accuracy_score:
            fast = lambda yt, idx: (yt == y_pred[idx]).mean(axis=1)
        elif metric_func is balanced_accuracy_score and y_pred.ndim == 1:
            labels = np.unique(np.concatenate([y_true, y_pred]))
            fast = lambda yt, idx: _balanced_accuracy_rows(yt, y_pred[idx], labels)
        elif y_pred.ndim == 1 and _is_binary_01(y_true):
            if metric_func is roc_auc_score:
                fast = lambda yt, idx: _auc_rows(yt == 1, y_pred[idx].astype(float))
            elif metric_func is average_precision_score:
                fast = lambda yt, idx: _average_precision_rows(yt == 1, y_pred[idx].astype(float))

    if fast is not None:
        valid_scores = []
        for idx in _iter_bootstrap_blocks(rng, len(y_pred), n_bootstraps):
            yt = y_true[idx]
            valid = _has_two_classes(yt)
            if valid.any():
                valid_scores.append(fast(yt[valid], idx[valid]))
        valid_scores = np.concatenate(valid_scores) if valid_scores else np.array([])
        if valid_scores.size == 0:
            return (None, None)
        return _percentile_ci(valid_scores, alpha)

    # Generic fallback (e.g. multiclass OvO AUC): one metric call per resample
    indices_list = [rng.randint(0, len(y_pred), len(y_pred)) for _ in range(n_bootstraps)]

    def compute(indices):
        if len(np.unique(y_true[indices])) < 2:
            return None
        return metric_func(y_true[indices], y_pred[indices], **kwargs)

    bootstrapped_scores = [compute(idx) for idx in indices_list]

    valid_scores = [s for s in bootstrapped_scores if s is not None]
    if not valid_scores:
        return (None, None)

    return _percentile_ci(valid_scores, alpha)


def get_cri_metrics_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.05, n_jobs=-1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(42)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    blocks = []
    for idx in _iter_bootstrap_blocks(rng, len(y_pred), n_bootstraps):
        yt = y_true[idx]
        valid = _has_two_classes(yt)
        if valid.any():
            blocks.append(_cri_avg_rows(yt[valid], y_pred[idx[valid]], labels))

    if not blocks:
        return {k: (None, None) for k in ['f1_ci', 'precision_ci', 'recall_ci', 'specificity_ci']}

    boot_f1, boot_pre, boot_rec, boot_spe = np.concatenate(blocks, axis=1)

    return {
        'f1_ci': _percentile_ci(boot_f1, alpha),
        'precision_ci': _percentile_ci(boot_pre, alpha),
        'recall_ci': _percentile_ci(boot_rec, alpha),
        'specificity_ci': _percentile_ci(boot_spe, alpha)
    }


# -------------------------------------------------------------------------
# Calibration and Decision Curve Analysis (DCA) Helpers
# -------------------------------------------------------------------------
def get_calibration_metrics(y_true, y_prob):
    brier = brier_score_loss(y_true, y_prob)
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    val_logit = np.log(y_prob_clipped / (1 - y_prob_clipped))

    lr = LogisticRegression(penalty=None, solver='lbfgs')
    lr.fit(val_logit.reshape(-1, 1), y_true)

    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]

    return brier, intercept, slope


def calculate_net_benefit(y_true, y_prob, thresholds=np.arange(0.01, 1.0, 0.01)):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    above = y_prob[None, :] >= thresholds[:, None]
    tp = (above & (y_true == 1)[None, :]).sum(axis=1)
    fp = (above & (y_true == 0)[None, :]).sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        net_benefits = (tp / n) - (fp / n) * (thresholds / (1 - thresholds))
    net_benefits = np.where(thresholds == 1.0, 0.0, net_benefits)
    return thresholds, net_benefits


# =========================================================================
#  PAIRED STATISTICAL TESTS  (DeLong / McNemar / paired bootstrap PR-AUC)
#  + Holm-Bonferroni step-down correction
# =========================================================================
def holm_correction(pvals):
    """Holm-Bonferroni step-down adjusted p-values, returned in input order."""
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    if m == 0:
        return np.array([])
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)  # enforce monotonicity
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


# --- DeLong test (fast algorithm, Sun & Xu 2014) --------------------------
def _compute_midrank(x):
    # mid-ranks (average rank for ties), 1-based
    return rankdata(x, method='average')


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """predictions_sorted_transposed: shape (2, n) with the positive samples first."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, y_prob_1, y_prob_2):
    """
    Two-sided DeLong test comparing two correlated ROC-AUCs on the same samples.
    Returns (auc_1, auc_2), p_value, z_statistic.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob_1 = np.asarray(y_prob_1, dtype=float)
    y_prob_2 = np.asarray(y_prob_2, dtype=float)

    label_1_count = int(y_true.sum())
    if label_1_count == 0 or label_1_count == len(y_true):
        return (np.nan, np.nan), np.nan, np.nan

    order = (-y_true).argsort(kind='mergesort')  # positives (label 1) first
    preds = np.vstack((y_prob_1, y_prob_2))[:, order]
    try:
        aucs, delongcov = _fast_delong(preds, label_1_count)
        l = np.array([[1.0, -1.0]])
        var = float((l @ np.asarray(delongcov) @ l.T).item())
        if not np.isfinite(var) or var <= 0:
            # zero variance -> the two AUCs carry the same information (e.g. identical scores)
            if np.isclose(aucs[0], aucs[1]):
                return (float(aucs[0]), float(aucs[1])), 1.0, 0.0
            return (float(aucs[0]), float(aucs[1])), np.nan, np.nan
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = 2.0 * norm.sf(abs(z))
        return (float(aucs[0]), float(aucs[1])), float(p), float(z)
    except Exception:
        return (np.nan, np.nan), np.nan, np.nan


# --- McNemar test on accuracy -------------------------------------------------
def mcnemar_test(y_true, y_pred_1, y_pred_2, exact_threshold=25):
    """
    Paired McNemar test on correct/incorrect classification.
    Exact binomial test when the number of discordant pairs <= exact_threshold,
    otherwise chi-square with continuity correction.
    Returns p_value, statistic, b, c  where
        b = # samples model_1 correct & model_2 wrong
        c = # samples model_1 wrong  & model_2 correct
    """
    y_true = np.asarray(y_true)
    correct_1 = (np.asarray(y_pred_1) == y_true)
    correct_2 = (np.asarray(y_pred_2) == y_true)

    b = int(np.sum(correct_1 & ~correct_2))
    c = int(np.sum(~correct_1 & correct_2))
    n = b + c
    if n == 0:
        return 1.0, 0.0, b, c

    if n <= exact_threshold:
        p = 2.0 * binom.cdf(min(b, c), n, 0.5)
        p = float(min(p, 1.0))
        stat = float(min(b, c))
    else:
        stat = (abs(b - c) - 1.0) ** 2 / n
        p = float(chi2.sf(stat, 1))
    return p, stat, b, c


# --- Paired bootstrap on PR-AUC --------------------------------------------
def paired_bootstrap_pr_auc_test(y_true, y_prob_base, y_prob_model,
                                 n_bootstraps=BOOTSTRAP_N_RESAMPLES, seed=42):
    """
    Paired bootstrap test on PR-AUC (average precision).
    Delta = AP(model) - AP(baseline), evaluated on the SAME resampled indices.
    Returns observed_diff, p_value, (ci_low, ci_high) of the difference.
    p_value = 2 * min( P(Delta* <= 0), P(Delta* >= 0) ).
    """
    y_true = np.asarray(y_true)
    y_prob_base = np.asarray(y_prob_base, dtype=float)
    y_prob_model = np.asarray(y_prob_model, dtype=float)
    n = len(y_true)

    observed = (average_precision_score(y_true, y_prob_model)
                - average_precision_score(y_true, y_prob_base))

    rng = np.random.RandomState(seed)
    diffs = np.full(n_bootstraps, np.nan, dtype=float)
    if _is_binary_01(y_true):
        start = 0
        for idx in _iter_bootstrap_blocks(rng, n, n_bootstraps):
            yt = y_true[idx]
            valid = _has_two_classes(yt)
            if valid.any():
                pos, idx_v = yt[valid] == 1, idx[valid]
                block = np.full(len(idx), np.nan)
                block[valid] = (_average_precision_rows(pos, y_prob_model[idx_v])
                                - _average_precision_rows(pos, y_prob_base[idx_v]))
                diffs[start:start + len(idx)] = block
            start += len(idx)
    else:
        for i in range(n_bootstraps):
            idx = rng.randint(0, n, n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            diffs[i] = (average_precision_score(y_true[idx], y_prob_model[idx])
                        - average_precision_score(y_true[idx], y_prob_base[idx]))

    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float(observed), np.nan, (np.nan, np.nan)

    p = 2.0 * min(np.mean(diffs <= 0.0), np.mean(diffs >= 0.0))
    p = float(min(p, 1.0))
    ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    return float(observed), p, ci


# -------------------------------------------------------------------------
# Pre-Computation Engine
# Folds are independent and each fold writes its own files, so they are
# computed in parallel worker processes (no file race conditions).
# -------------------------------------------------------------------------
def _precompute_single_fold(X_train, X_test, y_train, y_test, fold_num, filename_fs, feat_name, feat_type):
    X_train_file = os.path.join(filename_fs, f'X_train_fold_{fold_num}.csv')
    X_test_file = os.path.join(filename_fs, f'X_test_fold_{fold_num}.csv')
    y_train_file = os.path.join(filename_fs, f'y_train_fold_{fold_num}.npy')
    y_test_file = os.path.join(filename_fs, f'y_test_fold_{fold_num}.npy')

    if X_train.isnull().values.any():
        imp = SimpleImputer(strategy='most_frequent')
        X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(imp.transform(X_test), columns=X_test.columns, index=X_test.index)

    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    if feat_type == 'mrmr':
        # n_jobs=1: the folds themselves already run in parallel
        selected = mrmr_classif(X=X_train, y=y_train, K=5, return_scores=True, n_jobs=1, show_progress=False)
        top_genes = np.array(selected[1].sort_values().index)[-feat_name:]
        X_train, X_test = X_train[top_genes], X_test[top_genes]

    elif feat_type == 'lasso':
        lrcv = LinearSVC(C=5000, penalty="l1", dual=False).fit(X_train, y_train)
        top_genes = X_train.columns[
            np.where(SelectFromModel(lrcv, max_features=feat_name, prefit=True).get_support())[0]]
        X_train, X_test = X_train[top_genes], X_test[top_genes]

    elif feat_type == 'pca':
        pca = PCA(n_components=0.96)
        X_train = pd.DataFrame(pca.fit_transform(X_train), index=X_train.index)
        X_train.columns = [f'PCA_{i}' for i in range(X_train.shape[1])]
        X_test = pd.DataFrame(pca.transform(X_test), index=X_test.index, columns=X_train.columns)

    # y files first, then the X files (the "fold already computed" markers checked on reruns),
    # each via temp file + rename, so a run killed mid-write never leaves a half-written fold
    np.save(y_train_file, y_train)
    np.save(y_test_file, y_test)
    for df, out_file in ((X_test, X_test_file), (X_train, X_train_file)):
        tmp_file = f'{out_file}.{os.getpid()}.tmp'
        df.to_csv(tmp_file)
        os.replace(tmp_file, out_file)

    # the final selected feature group for this fold
    return fold_num, list(X_train.columns)


def precompute_folds_and_features(X, y, path, feat_name, k_fold_num, feat_type, n_jobs=N_JOBS):
    filename_fs = os.path.join(path, 'Results', 'Data_driven', 'Precomputed_Folds', f"{feat_type}_{feat_name}")
    os.makedirs(filename_fs, exist_ok=True)

    kf = StratifiedKFold(n_splits=k_fold_num) if k_fold_num > 1 else LeaveOneOut()
    n_splits = kf.get_n_splits(X)

    # Dictionary to track selected features per fold
    fold_features = {}
    todo = []

    for fold_num, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
        X_train_file = os.path.join(filename_fs, f'X_train_fold_{fold_num}.csv')
        X_test_file = os.path.join(filename_fs, f'X_test_fold_{fold_num}.csv')

        # Skip logic - load existing columns if already precomputed so tracking works on reruns
        if os.path.exists(X_train_file) and os.path.exists(X_test_file):
            fold_features[fold_num] = list(pd.read_csv(X_train_file, index_col=0, nrows=0).columns)
            continue

        todo.append(delayed(_precompute_single_fold)(
            X.iloc[train_index].copy(), X.iloc[test_index].copy(),
            y.iloc[train_index].values.ravel(), y.iloc[test_index].values.ravel(),
            fold_num, filename_fs, feat_name, feat_type))

    if todo:
        for fold_num, features in Parallel(n_jobs=n_jobs)(todo):
            fold_features[fold_num] = features

    fold_features = dict(sorted(fold_features.items()))

    # Save the feature tracking log to disk
    with open(os.path.join(filename_fs, 'selected_features_log.pkl'), 'wb') as f:
        pickle.dump(fold_features, f)

    return n_splits


from collections import Counter


def calculate_feature_stability(filename_fs):
    log_path = os.path.join(filename_fs, 'selected_features_log.pkl')
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'rb') as f:
        fold_features = pickle.load(f)

    # |C*| (Group Size) for each fold
    group_sizes = {f"fold_{fold}": len(features) for fold, features in fold_features.items()}
    avg_c_star = float(np.mean(list(group_sizes.values()))) if group_sizes else np.nan

    # Selection Frequency across all outer folds
    all_features = [feat for features in fold_features.values() for feat in features]
    frequencies = dict(Counter(all_features))

    # Sort features by highest selection frequency
    freq_counts = {feat: count for feat, count in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}

    return {
        'group_size_C_star_per_fold': group_sizes,
        'average_group_size_C_star': avg_c_star,
        'selection_frequency': freq_counts,
        'raw_fold_features': fold_features
    }


# -------------------------------------------------------------------------
# Per-fold metric computation (binary or multiclass)
# -------------------------------------------------------------------------
def _fold_classification_metrics(y_test, y_pred, y_prob, num_classes):
    """
    Metrics for a SINGLE fold's held-out set. Anything that is undefined for
    that fold (e.g. AUC when the fold's test set has only one class present,
    which can happen with small/imbalanced folds) is returned as NaN instead
    of raising, so one bad fold does not break the whole export.
    """
    metrics = {
        'acc': np.nan, 'balanced_acc': np.nan, 'f1': np.nan, 'precision': np.nan, 'recall': np.nan,
        'specificity': np.nan,
        'AUC': np.nan, 'PR_AUC': np.nan,
        'brier_score': np.nan, 'calib_intercept': np.nan, 'calib_slope': np.nan,
    }

    try:
        metrics['acc'] = accuracy_score(y_test, y_pred)
    except Exception:
        pass

    try:
        metrics['balanced_acc'] = balanced_accuracy_score(y_test, y_pred)
    except Exception:
        pass

    try:
        cri = classification_report_imbalanced(y_test, y_pred, output_dict=True)
        metrics['f1'] = cri['avg_f1']
        metrics['precision'] = cri['avg_pre']
        metrics['recall'] = cri['avg_rec']
        metrics['specificity'] = cri['avg_spe']
    except Exception:
        pass

    if len(np.unique(y_test)) < 2:
        # AUC / PR-AUC / calibration are undefined with a single class in the fold
        return metrics

    if num_classes == 2:
        try:
            metrics['AUC'] = roc_auc_score(y_test, y_prob[:, 1])
        except Exception:
            pass
        try:
            metrics['PR_AUC'] = average_precision_score(y_test, y_prob[:, 1])
        except Exception:
            pass
        try:
            brier, intercept, slope = get_calibration_metrics(y_test, y_prob[:, 1])
            metrics['brier_score'] = brier
            metrics['calib_intercept'] = intercept
            metrics['calib_slope'] = slope
        except Exception:
            pass
    else:
        try:
            metrics['AUC'] = roc_auc_score(y_test, y_prob, average='weighted', multi_class='ovo')
        except Exception:
            pass
        try:
            Y_bin = label_binarize(y_test, classes=np.unique(y_test))
            metrics['PR_AUC'] = average_precision_score(Y_bin, y_prob, average='weighted')
        except Exception:
            pass

    return metrics


# -------------------------------------------------------------------------
# Faster grid search for SVC(probability=True)
# -------------------------------------------------------------------------
class ProbaRefitGridSearchCV(GridSearchCV):
    """GridSearchCV for estimators with `probability=True` (SVC).

    With probability=True every SVC fit runs an extra internal 5-fold CV for Platt
    scaling, which makes each fit ~6x slower. The grid-search scorers used here
    ('roc_auc' -> decision_function, 'f1_weighted' -> predict) never use those
    probabilities, so the search is run with probability=False (same scores, same
    best_params_) and only the final best model is refit with probability=True
    so that predict_proba is available exactly as before."""

    def fit(self, X, y=None, **params):
        estimator = self.estimator
        self.estimator = clone(estimator).set_params(probability=False)
        try:
            super().fit(X, y, **params)
        finally:
            self.estimator = estimator
        if self.refit:
            self.best_estimator_ = clone(estimator).set_params(**self.best_params_).fit(X, y, **params)
        return self


def _make_grid_search(classifier, param_grid, metric_score):
    gs_cls = (ProbaRefitGridSearchCV
              if isinstance(classifier, SVC) and classifier.get_params().get('probability') else GridSearchCV)
    return gs_cls(classifier, param_grid, n_jobs=1, pre_dispatch='2*n_jobs', cv=StratifiedKFold(n_splits=5),
                  scoring=metric_score)


# -------------------------------------------------------------------------
# K-Fold Evaluation Engine (Only Reads Disk, No Processing)
# -------------------------------------------------------------------------
def _fit_predict_single_fold(name, classifier, param_grid, path, smote_flag, feat_name, fold_num, n_splits,
                             feat_type, num_classes):
    """Fit (or load the cached model for) ONE (classifier, outer fold) pair and predict its
    held-out set. Independent of every other (classifier, fold) pair and writes only its own
    model file, so these run safely in parallel worker processes."""
    filename_fs = os.path.join(path, 'Results', 'Data_driven', 'Precomputed_Folds', f"{feat_type}_{feat_name}")
    filename_model = os.path.join(path, 'Models', name, feat_type, str(feat_name))
    os.makedirs(filename_model, exist_ok=True)

    metric_score = 'roc_auc' if num_classes == 2 else 'f1_weighted'
    k_neigh = 3 if num_classes == 2 else 4

    # Load Precomputed Folds
    X_train_file = os.path.join(filename_fs, f'X_train_fold_{fold_num}.csv')
    X_test_file = os.path.join(filename_fs, f'X_test_fold_{fold_num}.csv')
    y_train_file = os.path.join(filename_fs, f'y_train_fold_{fold_num}.npy')
    y_test_file = os.path.join(filename_fs, f'y_test_fold_{fold_num}.npy')

    X_test = pd.read_csv(X_test_file, index_col=0)
    y_train = np.load(y_train_file)
    y_test = np.load(y_test_file)

    model_name = os.path.join(filename_model, f'model_{fold_num}_smote_{smote_flag}.pkl')

    # Fit or Load
    if not os.path.isfile(model_name):
        X_train = pd.read_csv(X_train_file, index_col=0)
        pipeline = Pipeline([
            ('smote', SMOTE(k_neighbors=k_neigh)) if smote_flag == 1 else ('passthrough', 'passthrough'),
            (name, _make_grid_search(classifier, param_grid, metric_score))
        ])
        pipeline.fit(X_train, y_train)
        # write to a temp file then rename, so an interrupted run never leaves a truncated pickle
        tmp_name = f'{model_name}.{os.getpid()}.tmp'
        with open(tmp_name, 'wb') as f:
            pickle.dump(pipeline, f)
        os.replace(tmp_name, model_name)
    else:
        with open(model_name, 'rb') as f:
            pipeline = pickle.load(f)

    y_pred_fold = pipeline.predict(X_test)
    y_prob_fold = pipeline.predict_proba(X_test)

    # --- per-fold metrics, computed on this fold's held-out set only ---
    fm = _fold_classification_metrics(y_test, y_pred_fold, y_prob_fold, num_classes)
    fm['fold'] = fold_num
    fm['n_train_samples'] = len(y_train)
    fm['n_test_samples'] = len(y_test)
    fm['n_features'] = X_test.shape[1]

    return name, fold_num, fm, y_test, y_pred_fold, y_prob_fold


def _pool_out_of_fold_results(fold_outputs, num_classes):
    """Pooled (out-of-fold) metrics + bootstrap CIs from the per-fold outputs of one classifier."""
    fold_outputs = sorted(fold_outputs, key=lambda o: o[1])  # by fold number
    fold_metrics_list = [o[2] for o in fold_outputs]
    oof_y_true = np.concatenate([np.asarray(o[3]) for o in fold_outputs])
    oof_y_pred = np.concatenate([np.asarray(o[4]) for o in fold_outputs])
    oof_y_prob = np.concatenate([np.asarray(o[5]) for o in fold_outputs])

    cri = classification_report_imbalanced(oof_y_true, oof_y_pred, output_dict=True)
    cri_cis = get_cri_metrics_ci(oof_y_true, oof_y_pred, n_jobs=-1)

    brier, calib_intercept, calib_slope, thresholds, net_benefits = None, None, None, None, None

    if num_classes == 2:
        auc_val = roc_auc_score(oof_y_true, oof_y_prob[:, 1])
        auc_ci = get_bootstrap_ci(oof_y_true, oof_y_prob[:, 1], roc_auc_score, n_jobs=-1)
        pr_auc_val = average_precision_score(oof_y_true, oof_y_prob[:, 1])
        pr_auc_ci = get_bootstrap_ci(oof_y_true, oof_y_prob[:, 1], average_precision_score, n_jobs=-1)
        brier, calib_intercept, calib_slope = get_calibration_metrics(oof_y_true, oof_y_prob[:, 1])
        thresholds, net_benefits = calculate_net_benefit(oof_y_true, oof_y_prob[:, 1])
    else:
        auc_val = roc_auc_score(oof_y_true, oof_y_prob, average='weighted', multi_class='ovo')
        auc_ci = get_bootstrap_ci(oof_y_true, oof_y_prob, roc_auc_score, average='weighted', multi_class='ovo',
                                  n_jobs=-1)
        Y_bin = label_binarize(oof_y_true, classes=np.unique(oof_y_true))
        pr_auc_val = average_precision_score(Y_bin, oof_y_prob, average='weighted')
        pr_auc_ci = (None, None)

    return {
        'acc': accuracy_score(oof_y_true, oof_y_pred),
        'acc_ci': get_bootstrap_ci(oof_y_true, oof_y_pred, accuracy_score, n_jobs=-1),
        'balanced_acc': balanced_accuracy_score(oof_y_true, oof_y_pred),
        'balanced_acc_ci': get_bootstrap_ci(oof_y_true, oof_y_pred, balanced_accuracy_score, n_jobs=-1),
        'f1': cri['avg_f1'], 'f1_ci': cri_cis['f1_ci'],
        'precision': cri['avg_pre'], 'precision_ci': cri_cis['precision_ci'],
        'recall': cri['avg_rec'], 'recall_ci': cri_cis['recall_ci'],
        'specificity': cri['avg_spe'], 'specificity_ci': cri_cis['specificity_ci'],
        'AUC': auc_val, 'AUC_ci': auc_ci,
        'PR_AUC': pr_auc_val, 'PR_AUC_ci': pr_auc_ci,
        'brier_score': brier,
        'calib_intercept': calib_intercept,
        'calib_slope': calib_slope,
        'dca_thresholds': thresholds,
        'dca_net_benefits': net_benefits,
        'fold_metrics': fold_metrics_list,
        'raw_y_true': oof_y_true,
        'raw_y_pred': oof_y_pred,
        'raw_y_prob': oof_y_prob
    }


def evaluate_single_model_kfold(name, classifier, param_grid, path, smote_flag, feat_name, n_splits, feat_type,
                                num_classes, n_jobs=N_JOBS):
    """Evaluate ONE classifier over all outer folds (folds fitted in parallel)."""
    fold_outputs = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_single_fold)(name, classifier, param_grid, path, smote_flag, feat_name,
                                          fold_num, n_splits, feat_type, num_classes)
        for fold_num in range(1, n_splits + 1))
    return name, _pool_out_of_fold_results(fold_outputs, num_classes)


# -------------------------------------------------------------------------
# Excel export builders
#
# `results_all` is expected to be {config_name: compiled_results} where
# compiled_results is the dict returned by traditional_ml_kfold (i.e.
# {'Feature_Selection_Stability': {...}, 'RandomForest': results_dict, ...}).
# -------------------------------------------------------------------------

# ---- Sheet 1: per-fold metrics (+ which features were selected in that fold) ----
def build_fold_metrics_df(results_all, smote_flag):
    """Long-format table: one row per (config, model, fold) with that fold's metrics
    and the feature/gene set selected for that fold (same set for every model within
    a config, since feature selection happens once per config before classifiers run)."""
    rows = []
    for config_name, compiled in results_all.items():
        if not isinstance(compiled, dict):
            continue
        fs = compiled.get('Feature_Selection_Stability') or {}
        fold_features = fs.get('raw_fold_features', {})

        for model_name, m in compiled.items():
            if model_name == 'Feature_Selection_Stability':
                continue
            if not isinstance(m, dict) or 'fold_metrics' not in m:
                continue
            for fm in m['fold_metrics']:
                feats = fold_features.get(fm['fold'], [])
                rows.append({
                    'smote': smote_flag,
                    'config': config_name,
                    'model': model_name,
                    'fold': fm['fold'],
                    'n_train_samples': fm['n_train_samples'],
                    'n_test_samples': fm['n_test_samples'],
                    'n_features': fm['n_features'],
                    'selected_features': ', '.join(map(str, feats)),
                    'accuracy': fm['acc'],
                    'balanced_accuracy': fm.get('balanced_acc', np.nan),
                    'AUC': fm['AUC'],
                    'PR_AUC': fm['PR_AUC'],
                    'F1': fm['f1'],
                    'precision': fm['precision'],
                    'recall': fm['recall'],
                    'specificity': fm['specificity'],
                    'brier_score': fm['brier_score'],
                    'calib_intercept': fm['calib_intercept'],
                    'calib_slope': fm['calib_slope'],
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['config', 'model', 'fold']).reset_index(drop=True)
    return df


# ---- Sheet 2: pooled (out-of-fold) average performance, with 95% bootstrap CIs ----
def _ci_str(ci):
    if ci is None or (isinstance(ci, (list, tuple)) and (len(ci) < 2 or ci[0] is None)):
        return ""
    try:
        return f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    except Exception:
        return ""


def build_pooled_performance_df(results_all, smote_flag):
    """One row per (config, model): performance pooled across all outer folds
    (i.e. computed once on the concatenated out-of-fold predictions), with the
    95% bootstrap CI already computed by evaluate_single_model_kfold."""
    rows = []
    for config_name, compiled in results_all.items():
        if not isinstance(compiled, dict):
            continue
        for model_name, m in compiled.items():
            if model_name == 'Feature_Selection_Stability':
                continue
            if not isinstance(m, dict) or 'acc' not in m:
                continue
            rows.append({
                'smote': smote_flag,
                'config': config_name,
                'model': model_name,
                'accuracy': m['acc'], 'accuracy_95CI': _ci_str(m.get('acc_ci')),
                'balanced_accuracy': m.get('balanced_acc', np.nan),
                'balanced_accuracy_95CI': _ci_str(m.get('balanced_acc_ci')),
                'AUC': m['AUC'], 'AUC_95CI': _ci_str(m.get('AUC_ci')),
                'PR_AUC': m['PR_AUC'], 'PR_AUC_95CI': _ci_str(m.get('PR_AUC_ci')),
                'F1': m['f1'], 'F1_95CI': _ci_str(m.get('f1_ci')),
                'precision': m['precision'], 'precision_95CI': _ci_str(m.get('precision_ci')),
                'recall': m['recall'], 'recall_95CI': _ci_str(m.get('recall_ci')),
                'specificity': m['specificity'], 'specificity_95CI': _ci_str(m.get('specificity_ci')),
                'brier_score': m.get('brier_score'),
                'calib_intercept': m.get('calib_intercept'),
                'calib_slope': m.get('calib_slope'),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['config', 'model']).reset_index(drop=True)
    return df


# ---- Shared pairwise-test machinery (used by Sheet 3 and Sheet 5) ----
def _compare_two_predictions(y_true, y_pred_a, y_prob_a, y_pred_b, y_prob_b,
                             n_bootstraps=BOOTSTRAP_N_RESAMPLES):
    """Paired DeLong (AUC) / McNemar (accuracy) / bootstrap (PR-AUC) between two sets of
    predictions on the SAME samples in the SAME order. Binary classification only
    (uses the positive-class probability column, index 1)."""
    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a, dtype=float)
    y_prob_b = np.asarray(y_prob_b, dtype=float)
    prob_a = y_prob_a[:, 1] if y_prob_a.ndim == 2 else y_prob_a
    prob_b = y_prob_b[:, 1] if y_prob_b.ndim == 2 else y_prob_b

    entry = {}

    (auc_a, auc_b), p_dl, z_dl = delong_roc_test(y_true, prob_a, prob_b)
    entry['delong'] = {'statistic': z_dl, 'p_raw': p_dl, 'auc_a': auc_a, 'auc_b': auc_b}

    p_mc, stat_mc, b_mc, c_mc = mcnemar_test(y_true, y_pred_a, y_pred_b)
    entry['mcnemar'] = {'statistic': stat_mc, 'p_raw': p_mc,
                        'a_correct_b_wrong': int(b_mc), 'b_correct_a_wrong': int(c_mc)}

    obs, p_bs, ci = paired_bootstrap_pr_auc_test(y_true, prob_a, prob_b, n_bootstraps=n_bootstraps)
    entry['bootstrap_pr_auc'] = {'statistic': obs, 'observed_diff': obs, 'p_raw': p_bs,
                                 'ci_low': ci[0], 'ci_high': ci[1], 'n_resamples': n_bootstraps}
    return entry


def _flip_pairwise_entry(entry):
    """Reverse a/b orientation for a comparison that was cached the other way round
    (p-values are symmetric and untouched; directional quantities get negated/swapped)."""
    out = {}
    for test, d in entry.items():
        d = dict(d)
        if test == 'delong':
            d['auc_a'], d['auc_b'] = d.get('auc_b'), d.get('auc_a')
            if d.get('statistic') is not None and np.isfinite(d['statistic']):
                d['statistic'] = -d['statistic']
        elif test == 'mcnemar':
            d['a_correct_b_wrong'], d['b_correct_a_wrong'] = d.get('b_correct_a_wrong'), d.get('a_correct_b_wrong')
        elif test == 'bootstrap_pr_auc':
            if d.get('statistic') is not None and np.isfinite(d['statistic']):
                d['statistic'] = -d['statistic']
            if d.get('observed_diff') is not None:
                d['observed_diff'] = -d['observed_diff']
            lo, hi = d.get('ci_low'), d.get('ci_high')
            d['ci_low'] = None if hi is None else -hi
            d['ci_high'] = None if lo is None else -lo
        out[test] = d
    return out


# ---- Sheet 3: classifiers compared pairwise WITHIN each feature-selection config ----
def run_classifier_pairwise_tests(compiled_results, baseline_models=BASELINE_MODELS,
                                  n_bootstraps=BOOTSTRAP_N_RESAMPLES):
    """DeLong / McNemar / bootstrap-PR-AUC between classifiers, using each classifier's
    pooled out-of-fold predictions within ONE config. Holm-corrected within each
    baseline's family (one baseline vs every other classifier), per test."""
    model_names = [k for k, v in compiled_results.items()
                   if isinstance(v, dict) and 'raw_y_prob' in v]
    if not model_names:
        return {}

    if baseline_models is None or (isinstance(baseline_models, str) and baseline_models.lower() == 'all'):
        baselines = list(model_names)
    elif isinstance(baseline_models, str):
        baselines = [baseline_models] if baseline_models in model_names else []
    else:
        baselines = [b for b in baseline_models if b in model_names]
    if not baselines:
        return {}

    y_true = np.asarray(compiled_results[model_names[0]]['raw_y_true'])

    needed = set()
    for b in baselines:
        for o in model_names:
            if o != b:
                needed.add(tuple(sorted((b, o))))

    pair_raw = {}
    for m1, m2 in needed:
        pair_raw[(m1, m2)] = _compare_two_predictions(
            y_true,
            compiled_results[m1]['raw_y_pred'], compiled_results[m1]['raw_y_prob'],
            compiled_results[m2]['raw_y_pred'], compiled_results[m2]['raw_y_prob'],
            n_bootstraps=n_bootstraps)

    results = {}
    for b in baselines:
        for o in model_names:
            if o == b:
                continue
            key = tuple(sorted((b, o)))
            entry = pair_raw[key]
            if key[0] != b:
                entry = _flip_pairwise_entry(entry)
            results[f"{b} vs {o}"] = entry

    for b in baselines:
        prefix = f"{b} vs "
        for test in _PAIRWISE_TEST_NAMES:
            keys = [k for k in results if k.startswith(prefix) and test in results[k]
                    and np.isfinite(results[k][test].get('p_raw', np.nan))]
            if not keys:
                continue
            adj = holm_correction([results[k][test]['p_raw'] for k in keys])
            for k, a in zip(keys, adj):
                results[k][test]['p_holm_within_config'] = float(a)

    return results


def build_classifier_pairwise_tests_df(results_all, smote_flag, baseline_models=BASELINE_MODELS,
                                       n_bootstraps=BOOTSTRAP_N_RESAMPLES, n_jobs=N_JOBS):
    configs = [(name, compiled) for name, compiled in results_all.items() if isinstance(compiled, dict)]
    # configs are independent -> run their pairwise tests in parallel
    all_sig = Parallel(n_jobs=n_jobs)(
        delayed(run_classifier_pairwise_tests)(compiled, baseline_models=baseline_models, n_bootstraps=n_bootstraps)
        for _, compiled in configs)

    rows = []
    for (config_name, _), sig in zip(configs, all_sig):
        for comp_name, entry in sig.items():
            baseline, _, model = comp_name.partition(' vs ')
            for test_name, d in entry.items():
                rows.append({
                    'smote': smote_flag,
                    'config': config_name,
                    'baseline': baseline,
                    'model': model,
                    'test': test_name,
                    'statistic': d.get('statistic', d.get('observed_diff', np.nan)),
                    'p_raw': d.get('p_raw', np.nan),
                    'p_holm_within_config': d.get('p_holm_within_config', np.nan),
                    'auc_baseline': d.get('auc_a', np.nan),
                    'auc_model': d.get('auc_b', np.nan),
                    'prauc_observed_diff': d.get('observed_diff', np.nan),
                    'prauc_diff_ci_low': d.get('ci_low', np.nan),
                    'prauc_diff_ci_high': d.get('ci_high', np.nan),
                    'mcnemar_baseline_correct_model_wrong': d.get('a_correct_b_wrong', np.nan),
                    'mcnemar_model_correct_baseline_wrong': d.get('b_correct_a_wrong', np.nan),
                    'n_resamples': d.get('n_resamples', np.nan),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['config', 'test', 'baseline', 'model']).reset_index(drop=True)
    return df


# ---- Sheet 4: decision-curve net benefit per threshold + pooled calibration ----
def build_dca_calibration_df(results_all, smote_flag):
    rows = []
    for config_name, compiled in results_all.items():
        if not isinstance(compiled, dict):
            continue
        for model_name, m in compiled.items():
            if model_name == 'Feature_Selection_Stability':
                continue
            if not isinstance(m, dict) or m.get('dca_thresholds') is None:
                continue
            for pt, nb in zip(m['dca_thresholds'], m['dca_net_benefits']):
                rows.append({
                    'smote': smote_flag,
                    'config': config_name,
                    'model': model_name,
                    'threshold': float(pt),
                    'net_benefit': float(nb),
                    'brier_score': m.get('brier_score'),
                    'calib_intercept': m.get('calib_intercept'),
                    'calib_slope': m.get('calib_slope'),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['config', 'model', 'threshold']).reset_index(drop=True)
    return df


# ---- Sheet 5: feature-selection-type comparisons, each side via its own best classifier ----
def _fs_type_of(config_name):
    for fs_type in FEATURE_SELECTION_TYPES:
        if config_name.startswith(fs_type):
            return fs_type
    return None


def select_best_per_fs_type(results_all, metric=FS_TYPE_SELECTION_METRIC):
    """For each feature-selection TYPE, find the single (config, classifier) pair with the
    highest pooled `metric` across every config of that type and every classifier."""
    best = {}
    for config_name, compiled in results_all.items():
        if not isinstance(compiled, dict):
            continue
        fs_type = _fs_type_of(config_name)
        if fs_type is None:
            continue
        for model_name, m in compiled.items():
            if model_name == 'Feature_Selection_Stability':
                continue
            if not isinstance(m, dict) or metric not in m:
                continue
            val = m[metric]
            if val is None or not np.isfinite(val):
                continue
            current = best.get(fs_type)
            if current is None or val > current['metric_value']:
                best[fs_type] = {
                    'config': config_name, 'model': model_name, 'metric_value': val,
                    'raw_y_true': m['raw_y_true'], 'raw_y_pred': m['raw_y_pred'], 'raw_y_prob': m['raw_y_prob'],
                }
    return best


def build_fs_type_comparison_df(results_all, smote_flag, n_bootstraps=BOOTSTRAP_N_RESAMPLES):
    """The FEATURE_SELECTION_COMPARISON_PAIRS comparisons. Each type is represented by its
    single best (config, classifier) pair (highest pooled FS_TYPE_SELECTION_METRIC). A pair
    is skipped (with a note) if either side has no results, or if the two configs' pooled
    y_true arrays don't line up sample-for-sample (so the paired tests would be invalid)."""
    best = select_best_per_fs_type(results_all)
    rows = []
    p_raw_by_test = {t: [] for t in _PAIRWISE_TEST_NAMES}
    row_idx_by_test = {t: [] for t in _PAIRWISE_TEST_NAMES}

    for type_a, type_b in FEATURE_SELECTION_COMPARISON_PAIRS:
        comparison = f"{type_a} vs {type_b}"
        info_a, info_b = best.get(type_a), best.get(type_b)
        if info_a is None or info_b is None:
            missing = type_a if info_a is None else type_b
            rows.append({'smote': smote_flag, 'comparison': comparison, 'type_a': type_a, 'type_b': type_b,
                         'note': f"skipped: no results found for '{missing}'"})
            continue

        y_true_a = np.asarray(info_a['raw_y_true'])
        y_true_b = np.asarray(info_b['raw_y_true'])
        if y_true_a.shape != y_true_b.shape or not np.array_equal(y_true_a, y_true_b):
            rows.append({'smote': smote_flag, 'comparison': comparison, 'type_a': type_a, 'type_b': type_b,
                         'note': "skipped: outer-fold label order differs between configs - not a valid paired comparison"})
            continue

        entry = _compare_two_predictions(
            y_true_a, info_a['raw_y_pred'], info_a['raw_y_prob'],
            info_b['raw_y_pred'], info_b['raw_y_prob'], n_bootstraps=n_bootstraps)

        for test_name, d in entry.items():
            rows.append({
                'smote': smote_flag,
                'comparison': comparison,
                'type_a': type_a, 'type_a_config': info_a['config'], 'type_a_model': info_a['model'],
                f'type_a_{FS_TYPE_SELECTION_METRIC}': info_a['metric_value'],
                'type_b': type_b, 'type_b_config': info_b['config'], 'type_b_model': info_b['model'],
                f'type_b_{FS_TYPE_SELECTION_METRIC}': info_b['metric_value'],
                'test': test_name,
                'statistic': d.get('statistic', d.get('observed_diff', np.nan)),
                'p_raw': d.get('p_raw', np.nan),
                'auc_type_a': d.get('auc_a', np.nan),
                'auc_type_b': d.get('auc_b', np.nan),
                'prauc_observed_diff': d.get('observed_diff', np.nan),
                'prauc_diff_ci_low': d.get('ci_low', np.nan),
                'prauc_diff_ci_high': d.get('ci_high', np.nan),
                'mcnemar_a_correct_b_wrong': d.get('a_correct_b_wrong', np.nan),
                'mcnemar_b_correct_a_wrong': d.get('b_correct_a_wrong', np.nan),
                'n_resamples': d.get('n_resamples', np.nan),
                'note': '',
            })
            if np.isfinite(d.get('p_raw', np.nan)):
                p_raw_by_test[test_name].append(d['p_raw'])
                row_idx_by_test[test_name].append(len(rows) - 1)

    # Holm-Bonferroni across the 12 comparisons, separately per test
    for test_name in _PAIRWISE_TEST_NAMES:
        idx = row_idx_by_test[test_name]
        if not idx:
            continue
        adj = holm_correction(p_raw_by_test[test_name])
        for i, a in zip(idx, adj):
            rows[i]['p_holm'] = float(a)

    return pd.DataFrame(rows)


# ---- Top-level export: writes all 5 sheets to one workbook ----
def export_results_to_excel(results_all, smote_flag, out_dir='.',
                            baseline_models=BASELINE_MODELS, n_bootstraps=BOOTSTRAP_N_RESAMPLES):
    """
    Write one .xlsx with:
      Sheet1 Fold_Metrics                - metrics + selected features for every (config, model, fold)
      Sheet2 Pooled_Performance          - pooled out-of-fold performance + 95% bootstrap CIs, per (config, model)
      Sheet3 Classifier_Pairwise_Tests   - DeLong (AUC) / McNemar (accuracy) / bootstrap (PR-AUC), classifiers
                                            compared within each config
      Sheet4 DCA_Calibration             - decision-curve net benefit per threshold + pooled calibration, per
                                            (config, model)
      Sheet5 FeatureSelection_Comparison - the 12 feature-selection-type comparisons, each side represented
                                            by its own best classifier (highest pooled FS_TYPE_SELECTION_METRIC)
    """
    tag = 'smote' if smote_flag == 1 else 'no_smote'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'ml_results_{tag}.xlsx')

    sheets = {
        'Fold_Metrics': build_fold_metrics_df(results_all, smote_flag),
        'Pooled_Performance': build_pooled_performance_df(results_all, smote_flag),
        'Classifier_Pairwise_Tests': build_classifier_pairwise_tests_df(
            results_all, smote_flag, baseline_models, n_bootstraps),
        'DCA_Calibration': build_dca_calibration_df(results_all, smote_flag),
        'FeatureSelection_Comparison': build_fs_type_comparison_df(results_all, smote_flag, n_bootstraps),
    }

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            if df.empty:
                df = pd.DataFrame({'info': ['no data']})
            elif len(df) > EXCEL_MAX_DATA_ROWS:
                # Too many rows for one Excel sheet (e.g. DCA_Calibration with ~2,000 configs x 9 models
                # x 99 thresholds): write the full table to a CSV next to the workbook and leave a pointer.
                csv_path = os.path.join(out_dir, f'ml_results_{tag}_{sheet_name}.csv')
                df.to_csv(csv_path, index=False)
                print(f"[Excel] sheet '{sheet_name}' has {len(df):,} rows (Excel max {EXCEL_MAX_DATA_ROWS:,}); "
                      f"wrote it to {os.path.abspath(csv_path)}")
                df = pd.DataFrame({'info': [f"{len(df):,} rows - too many for Excel, saved to "
                                            f"{os.path.basename(csv_path)} in the same folder"]})
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"[Excel] wrote {os.path.abspath(out_path)}")
    return out_path


# -------------------------------------------------------------------------
# The Main Caller Function
# -------------------------------------------------------------------------
def prepare_base_directories(path):
    dirs = [
        os.path.join(path, 'Models'),
        os.path.join(path, 'Results', 'Data_driven', 'Precomputed_Folds')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_classifiers():
    return {
        'SVM_linear': (SVC(probability=True, kernel='linear', class_weight='balanced', max_iter=10000),
                       {'C': [0.01, 0.1, 1, 10]}),
        'SVM_rbf': (SVC(probability=True, kernel='rbf', class_weight='balanced', max_iter=10000),
                    {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 0.5, 1, 2, 5]}),
        'RandomForest': (RandomForestClassifier(class_weight='balanced'), {'n_estimators': [50, 100, 200]}),
        'LogisticRegression': (LogisticRegression(max_iter=10000, class_weight='balanced'), {'C': [0.1, 1, 10]}),
        'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9]}),
        'GradientBoosting': (GradientBoostingClassifier(),
                             {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}),
        'AdaBoost': (AdaBoostClassifier(), {'n_estimators': [100, 200, 500]}),
        'ExtraTrees': (ExtraTreesClassifier(class_weight='balanced'),
                       {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}),
        'LinearDiscriminantAnalysis': (LinearDiscriminantAnalysis(), {}),
    }


# Rough relative cost of one fit, used only to schedule the slowest jobs first (better load balancing)
_CLASSIFIER_COST = {'GradientBoosting': 8, 'AdaBoost': 6, 'ExtraTrees': 3, 'RandomForest': 3, 'SVM_rbf': 2}


def _check_feature_group(X, feat_name):
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"Feature group {feat_name!r} is a {type(X).__name__}, expected a DataFrame holding that group's "
            f"genes as columns. If you looped over X.keys() of the expression DataFrame itself, you are "
            f"iterating over single genes: build the groups first, e.g. "
            f"X_groups = pathway_based_fs(X, pathway_list), and loop over X_groups.")


def traditional_ml_kfold(X, y, path, smote_flag, feat_name, k_fold_num, feat_type, n_jobs=N_JOBS,
                         show_progress=True):
    _check_feature_group(X, feat_name)
    prepare_base_directories(path)
    classifiers = get_classifiers()
    num_classes = len(pd.unique(y.values.ravel()))

    # --- 1. COMPUTE FOLDS & FEATURES ONCE (folds in parallel) ---
    n_splits = precompute_folds_and_features(X, y, path, feat_name, k_fold_num, feat_type, n_jobs=n_jobs)

    # Load and attach feature stability metrics (|C*| + selection frequency)
    filename_fs = os.path.join(path, 'Results', 'Data_driven', 'Precomputed_Folds', f"{feat_type}_{feat_name}")
    feature_stability_metrics = calculate_feature_stability(filename_fs)

    # --- 2. PARALLEL MODEL EVALUATION ---
    # Every (classifier, outer fold) pair is an independent job, so all 9 x n_splits of them
    # share one worker pool. Each GridSearchCV stays single-process (n_jobs=1) to avoid
    # nested parallelism / CPU oversubscription.
    order = sorted(classifiers, key=lambda n: -_CLASSIFIER_COST.get(n, 1) * len(ParameterGrid(classifiers[n][1])))
    jobs = [delayed(_fit_predict_single_fold)(name, classifiers[name][0], classifiers[name][1], path, smote_flag,
                                              feat_name, fold_num, n_splits, feat_type, num_classes)
            for name in order for fold_num in range(1, n_splits + 1)]

    outputs_by_model = {name: [] for name in classifiers}
    for out in tqdm(Parallel(n_jobs=n_jobs, return_as='generator_unordered')(jobs), total=len(jobs),
                    desc=f"Evaluating ({feat_type}:{feat_name})", disable=not show_progress):
        outputs_by_model[out[0]].append(out)

    compiled_results = {}
    compiled_results['Feature_Selection_Stability'] = feature_stability_metrics
    for name in classifiers:
        compiled_results[name] = _pool_out_of_fold_results(outputs_by_model[name], num_classes)

    return compiled_results


def _run_config_sequential(X, y, path, smote_flag, feat_name, k_fold_num, feat_type):
    try:
        return feat_name, traditional_ml_kfold(X, y, path, smote_flag, feat_name, k_fold_num, feat_type,
                                               n_jobs=1, show_progress=False), None
    except Exception as e:
        return feat_name, None, f"{type(e).__name__}: {e}"


def traditional_ml_kfold_many(X_dict, y, path, smote_flag, k_fold_num, feat_type, n_jobs=N_JOBS):
    """Run traditional_ml_kfold for MANY feature subsets (e.g. ~1,600 pathways) at once.

    With many small subsets it is faster to parallelise over the subsets: each worker runs
    one whole subset (fold pre-computation, all classifiers, bootstraps) sequentially, so
    the worker pool stays busy, nothing waits on the slowest classifier of a subset, and
    the per-subset setup/pooling work is parallel too. Every subset writes to its own
    directories, so there are no file races.

    A subset that raises is reported and left out instead of aborting the whole batch; its
    fitted models are cached on disk, so simply re-running resumes where it stopped.

    Returns {feat_name: compiled_results}, in the order of X_dict."""
    # fail fast, before any work is dispatched
    if not isinstance(X_dict, dict):
        raise TypeError(f"X_dict must be a dict {{group_name: DataFrame}}, got {type(X_dict).__name__}. "
                        f"Build it with pathway_based_fs / region_based_fs / chromosome_based_fs.")
    for feat_name, X_sub in X_dict.items():
        _check_feature_group(X_sub, feat_name)

    jobs = [delayed(_run_config_sequential)(X_sub, y, path, smote_flag, feat_name, k_fold_num, feat_type)
            for feat_name, X_sub in X_dict.items()]
    results, failed = {}, {}
    for feat_name, compiled, error in tqdm(Parallel(n_jobs=n_jobs, return_as='generator_unordered')(jobs),
                                           total=len(jobs), desc=f"{feat_type} (smote={smote_flag})"):
        if error is None:
            results[feat_name] = compiled
        else:
            failed[feat_name] = error
            print(f"[{feat_type}] '{feat_name}' failed: {error}", flush=True)
    if failed:
        print(f"[{feat_type}] {len(failed)}/{len(jobs)} subsets failed and were skipped "
              f"(first: {next(iter(failed))!r}: {next(iter(failed.values()))})", flush=True)
    return {feat_name: results[feat_name] for feat_name in X_dict if feat_name in results}


# -------------------------------------------------------------------------
# Example driver: sweep every feat_type/feat_name config, then run BOTH
# comparisons - classifiers-within-a-config (existing) and methods-against-
# each-other (new, the 18 comparisons).
# -------------------------------------------------------------------------
# In[2]:
def pathway_based_fs(X, pathway_data):
    new_X = {}
    x_columns = set(X.columns)
    for i in list(pathway_data.keys()):
        pathway_list = list(x_columns & set(pathway_data[i]))
        if pathway_list == []:
            continue
        new_X[i] = X[pathway_list].reindex(sorted(X[pathway_list].columns), axis=1)
    return new_X


def median_gene_data(X):
    gene_duplicated = np.unique(X.iloc[:, X.columns.duplicated()].columns)
    data_cleaned = []
    for i in gene_duplicated:
        data_cleaned.append(X[i].median(axis=1))
    if data_cleaned:
        data_cleaned_1 = pd.concat(data_cleaned, axis=1)
        data_cleaned_1.columns = gene_duplicated
        data_cleaned_2 = X.drop(columns=gene_duplicated)
        data_cleaned_3 = pd.concat([data_cleaned_1, data_cleaned_2], axis=1)
        data_cleaned_4 = data_cleaned_3.reindex(sorted(data_cleaned_3.columns), axis=1)
        return data_cleaned_4
    return X.reindex(sorted(X.columns), axis=1)


def region_based_fs(X, annotation_list):
    indx_annot, indx_data = indx_intersect_two_lists(list(annotation_list.hgnc_symbol), list(X.columns))
    annotation_list = annotation_list.iloc[indx_annot, :]
    X = X.iloc[:, indx_data]

    data_region_based = {}
    for band_id in pd.unique(annotation_list.band):
        a = annotation_list[annotation_list['band'] == band_id]
        for chr_name_band in sorted(pd.unique(a.chromosome_name)):
            b = a[a.chromosome_name == chr_name_band]
            list_gene_region = set(b['hgnc_symbol'])

            list_gene_region_ = [value for value in X.columns if value in list_gene_region]
            if list_gene_region_ == []:
                continue

            data_region_based['band ' + band_id + ', chr num ' + str(chr_name_band)] = X[list_gene_region_].reindex(
                sorted(X[list_gene_region_].columns), axis=1)
    return data_region_based


def chromosome_based_fs(X, annotation_list):
    indx_annot, indx_data = indx_intersect_two_lists(list(annotation_list.hgnc_symbol), list(X.columns))
    annotation_list = annotation_list.iloc[indx_annot, :]
    X = X.iloc[:, indx_data]

    data_chr_based = {}
    for chr_name_band in sorted(pd.unique(annotation_list.chromosome_name)):
        b = annotation_list[annotation_list.chromosome_name == chr_name_band]
        list_gene_region = set(b['hgnc_symbol'].values)

        list_gene_region_ = [value for value in X.columns if value in list_gene_region]
        if list_gene_region_ == []:
            continue
        data_chr_based['chr num ' + str(chr_name_band)] = X[list_gene_region_].reindex(
            sorted(X[list_gene_region_].columns), axis=1)
    return data_chr_based


def indx_intersect_two_lists(list_1, list_2):
    both = list(set(list_1).intersection(list_2))
    # first-occurrence index lookup in O(1) instead of list.index() (O(n) per gene)
    first_1, first_2 = {}, {}
    for i, v in enumerate(list_1):
        first_1.setdefault(v, i)
    for i, v in enumerate(list_2):
        first_2.setdefault(v, i)
    indices_A = [first_1[x] for x in both]
    indices_B = [first_2[x] for x in both]
    return indices_A, indices_B

# In[8]:
