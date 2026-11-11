"""
corrected_nested_cv.py
======================
Corrected nested cross-validation for the biology-informed feature-selection
paper, addressing Reviewer 3 comment R3.1.

WHAT CHANGED RELATIVE TO utils.py
---------------------------------
In the submitted analysis the pipeline was:

    for every group of the taxonomy:
        run all 9 classifiers through 10-fold CV
    then: select_best_per_fs_type(...)  ->  argmax of POOLED OUT-OF-FOLD AUC

i.e. both the biological group and the classifier were chosen by looking at the
held-out predictions they were then scored on.  The same applies to the number
of features of mRMR / LASSO / PCA.  That is the selection bias R3.1 describes.

Here, inside EVERY outer training fold and using ONLY that fold's training data:

  1. the ranking Phi of Eq. (3) scores every group of the taxonomy by the mean
     balanced accuracy of a logistic-regression model over the inner folds;
  2. the top-ranked group C* of Eq. (4) is retained;
  3. the classifier and its hyperparameters are chosen by the same inner CV;
  4. for the data-driven baselines the number of features / components is chosen
     by the same inner CV;
  5. for the k-means counterpart, genes are clustered by expression on the
     training part (k by the elbow method, no labels), and the clusters are
     ranked and selected with Phi exactly as in steps 1-2.

The outer test fold is touched exactly once, to produce the predictions that are
scored.

A NOTE ON WHAT MAY LEGITIMATELY BE BUILT OUTSIDE THE LOOP
--------------------------------------------------------
The group DEFINITIONS (which gene belongs to which chromosome, cytogenetic band
or Reactome pathway) come from external databases and carry no information about
the labels, so they are built once.  What must happen inside the fold is the
CHOICE of one group among them, and that is what Phi does here.

OUTPUTS
-------
For each cohort and each smote_flag, a pickle holding, per method:
  * oof_y_true / oof_y_pred / oof_y_prob   - per-sample out-of-fold predictions
                                             (needed for the paired tests and
                                             for the reliability curves, R3.5)
  * fold_group    - the group selected in each outer fold   (Table VII 'Sel.')
  * fold_model    - the classifier selected in each outer fold
  * fold_n_feat   - |C*| in each outer fold
  * pooled metrics with bootstrap 95% CIs and calibration

Python 3.9 compatible.  Reuses the project's own utils.py for the group
builders, the classifier zoo, calibration and the paired tests.
"""

import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (balanced_accuracy_score, accuracy_score, roc_auc_score,
                             average_precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from joblib import Parallel, delayed

from mrmr import mrmr_classif

from utils import utils


# =========================================================================
# Configuration
# =========================================================================
N_OUTER_FOLDS = 10           # as in the submitted analysis
N_INNER_FOLDS = 5            # as in the submitted analysis
RANDOM_STATE = 42            # fixed so the run is reproducible
N_JOBS = -1

# Feature counts offered to the data-driven baselines. The inner CV picks one
# per outer fold; it is no longer fixed by looking at outer-fold performance.
MRMR_LASSO_FEATURE_GRID = [50, 100, 200, 500, 1000, 2000, 5000]
PCA_VARIANCE_GRID = [0.95]

# NOTE (flagged for the authors): the submitted utils.py calls
#   mrmr_classif(X, y, K=5, return_scores=True)
# and then keeps the top `feat_name` genes by the RELEVANCE score, so for any
# feat_name > 5 the selection is an F-statistic ranking rather than a full
# redundancy-aware mRMR selection. The behaviour is reproduced here so the new
# numbers stay comparable with the submitted ones. Set MRMR_K_FULL = True to
# run mRMR properly instead -- but then say so in the manuscript.
MRMR_K_PARAM = 5
MRMR_K_FULL = False

BI_TAXONOMIES = ('chromosome_based', 'region_based', 'pathway_based')
BASELINES = ('mrmr', 'lasso', 'pca')

# k-means counterpart of the taxonomies: genes are clustered by expression on
# each outer training part, and the clusters are ranked and selected with the
# same Phi as the biological groups, so only the origin of the groups differs.
# k is the elbow of the within-cluster sum of squares over this grid.
KMEANS = 'kmeans'
KMEANS_K_GRID = [10, 25, 50, 100, 200, 400]

ALL_METHODS = BI_TAXONOMIES + BASELINES + (KMEANS,)


# =========================================================================
# Helpers
# =========================================================================
def _inner_cv(seed=RANDOM_STATE):
    return StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)


def _outer_cv(seed=RANDOM_STATE):
    return StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=seed)


def _impute_and_scale(X_train, X_test, study_train=None, study_test=None):
    """Imputer and scaler fitted on the training fold only.

    For a cohort pooled from several studies, each study's genes are
    standardised with that study's own training samples, and its test samples
    with the same scaler. This removes differences in level and spread between
    studies without using the labels or the test fold.
    """
    if X_train.isnull().values.any():
        imp = SimpleImputer(strategy='most_frequent')
        X_train = pd.DataFrame(imp.fit_transform(X_train),
                               columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(imp.transform(X_test),
                              columns=X_test.columns, index=X_test.index)
    if study_train is None:
        scaler = StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train),
                               index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test),
                              index=X_test.index, columns=X_test.columns)
        return X_train, X_test

    missing = set(np.unique(study_test)) - set(np.unique(study_train))
    if missing:
        raise ValueError('studies %s have test samples but no training samples' % sorted(missing))
    tr_vals = X_train.to_numpy(dtype=float, copy=True)
    te_vals = X_test.to_numpy(dtype=float, copy=True)
    for s in np.unique(study_train):
        tr, te = study_train == s, study_test == s
        scaler = StandardScaler().fit(tr_vals[tr])
        tr_vals[tr] = scaler.transform(tr_vals[tr])
        if te.any():
            te_vals[te] = scaler.transform(te_vals[te])
    return (pd.DataFrame(tr_vals, index=X_train.index, columns=X_train.columns),
            pd.DataFrame(te_vals, index=X_test.index, columns=X_test.columns))


def _group_columns(X_groups, X_columns):
    """{group name: [gene columns present in X]} for one taxonomy."""
    present = set(X_columns)
    out = {}
    for name, df in X_groups.items():
        cols = [c for c in df.columns if c in present]
        if cols:
            out[name] = cols
    return out


# =========================================================================
# (1) The ranking Phi  -- Eq. (3), computed on the training fold only
# =========================================================================
def _phi_one_group(X_tr_vals, y_tr, cols_idx, inner_splits):
    """Mean balanced accuracy of logistic regression over the inner folds."""
    Xg = X_tr_vals[:, cols_idx]
    scores = []
    for tr, va in inner_splits:
        if len(np.unique(y_tr[tr])) < 2 or len(np.unique(y_tr[va])) < 2:
            continue
        sc = StandardScaler().fit(Xg[tr])
        clf = LogisticRegression(max_iter=10000, class_weight='balanced')
        try:
            clf.fit(sc.transform(Xg[tr]), y_tr[tr])
            pred = clf.predict(sc.transform(Xg[va]))
            scores.append(balanced_accuracy_score(y_tr[va], pred))
        except Exception:
            continue
    return float(np.mean(scores)) if scores else np.nan


def rank_groups_phi(X_tr, y_tr, group_cols, n_jobs=N_JOBS):
    """Eq. (3): score every group. Eq. (4): the caller takes the argmax.

    Training-fold data only -- the outer test fold is not visible here.
    """
    col_pos = {c: i for i, c in enumerate(X_tr.columns)}
    X_vals = X_tr.values
    inner_splits = list(_inner_cv().split(X_vals, y_tr))

    names = list(group_cols.keys())
    idx_lists = [np.array([col_pos[c] for c in group_cols[n]], dtype=int) for n in names]

    scores = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_phi_one_group)(X_vals, y_tr, idx, inner_splits) for idx in idx_lists)

    return {n: s for n, s in zip(names, scores) if s == s}   # drop NaN


# =========================================================================
# (2) Classifier + hyperparameter choice -- inner CV of the SAME fold
# =========================================================================
def select_classifier_inner(X_tr, y_tr, smote_flag, n_jobs=N_JOBS):
    """Return (name, fitted_pipeline, inner_score) for the best of the 9 classifiers.

    SMOTE sits INSIDE the inner cross-validation, so the inner validation splits
    never contain synthetic samples. (The submitted code oversampled the whole
    training fold before the inner search; this is the stricter arrangement.)
    """
    classifiers = utils.get_classifiers()
    k_neigh = 3
    best = (None, None, -np.inf)

    for name, (clf, grid) in classifiers.items():
        if smote_flag == 1:
            pipe = Pipeline([('smote', SMOTE(k_neighbors=k_neigh, random_state=RANDOM_STATE)),
                             ('clf', clone(clf))])
            param_grid = {'clf__' + k: v for k, v in grid.items()}
        else:
            pipe = Pipeline([('clf', clone(clf))])
            param_grid = {'clf__' + k: v for k, v in grid.items()}

        try:
            gs = GridSearchCV(pipe, param_grid, cv=_inner_cv(), scoring='roc_auc',
                              n_jobs=n_jobs, error_score=np.nan)
            gs.fit(X_tr, y_tr)
        except Exception:
            continue

        score = gs.best_score_
        if score is not None and np.isfinite(score) and score > best[2]:
            best = (name, gs.best_estimator_, float(score))

    return best


# =========================================================================
# (3) Baseline feature-count choice -- inner CV of the SAME fold
# =========================================================================
def _rank_features_mrmr(X_in, y_in, k_max):
    """Ranked gene list, best first. Mirrors the submitted selector (see NOTE)."""
    k = min(len(X_in.columns), k_max) if MRMR_K_FULL else MRMR_K_PARAM
    selected = mrmr_classif(X=X_in, y=y_in, K=k, return_scores=True,
                            n_jobs=1, show_progress=False)
    if MRMR_K_FULL:
        return list(selected[0])
    # relevance scores, descending -- exactly what the submitted code ranks by
    return list(selected[1].sort_values(ascending=False).index)


def _rank_features_lasso(X_in, y_in, k_max):
    lrcv = LinearSVC(C=5000, penalty='l1', dual=False, max_iter=10000).fit(X_in, y_in)
    support = SelectFromModel(lrcv, max_features=k_max, prefit=True).get_support()
    chosen = X_in.columns[np.where(support)[0]]
    coef = np.abs(np.asarray(lrcv.coef_).ravel())
    order = np.argsort(coef)[::-1]
    ranked = [X_in.columns[i] for i in order if X_in.columns[i] in set(chosen)]
    return ranked


def select_baseline_config_inner(X_tr, y_tr, method, smote_flag):
    """Choose the feature/component count by inner CV on the training fold only.

    The selector is refitted inside every inner training split; the ranked list
    is produced once per split at the largest grid value and smaller values are
    read off as prefixes of it, which is identical to refitting per value for
    these rank-based selectors and far cheaper.
    """
    inner_splits = list(_inner_cv().split(X_tr.values, y_tr))

    if method == 'pca':
        grid = PCA_VARIANCE_GRID
    else:
        grid = [k for k in MRMR_LASSO_FEATURE_GRID if k <= X_tr.shape[1]] or [X_tr.shape[1]]

    scores = {g: [] for g in grid}

    for tr, va in inner_splits:
        X_in, X_va = X_tr.iloc[tr], X_tr.iloc[va]
        y_in, y_va = y_tr[tr], y_tr[va]
        if len(np.unique(y_in)) < 2 or len(np.unique(y_va)) < 2:
            continue

        if method == 'pca':
            for g in grid:
                try:
                    pca = PCA(n_components=g).fit(X_in)
                    Zi, Zv = pca.transform(X_in), pca.transform(X_va)
                    scores[g].append(_quick_auc(Zi, y_in, Zv, y_va, smote_flag))
                except Exception:
                    continue
            continue

        k_max = max(grid)
        try:
            ranked = (_rank_features_mrmr(X_in, y_in, k_max) if method == 'mrmr'
                      else _rank_features_lasso(X_in, y_in, k_max))
        except Exception:
            continue
        for g in grid:
            cols = ranked[:g]
            if not cols:
                continue
            try:
                scores[g].append(_quick_auc(X_in[cols].values, y_in,
                                            X_va[cols].values, y_va, smote_flag))
            except Exception:
                continue

    means = {g: (np.nanmean(v) if v else -np.inf) for g, v in scores.items()}
    best_g = max(means, key=means.get)
    return best_g


def _quick_auc(Z_tr, y_tr, Z_va, y_va, smote_flag):
    """Cheap proxy model used ONLY to compare feature counts inside a fold."""
    if smote_flag == 1:
        try:
            Z_tr, y_tr = SMOTE(k_neighbors=3, random_state=RANDOM_STATE).fit_resample(Z_tr, y_tr)
        except Exception:
            pass
    clf = LogisticRegression(max_iter=10000, class_weight='balanced')
    clf.fit(Z_tr, y_tr)
    return roc_auc_score(y_va, clf.predict_proba(Z_va)[:, 1])


# =========================================================================
# (3b) k-means groups -- the data-driven counterpart of the taxonomies
# =========================================================================
_KMEANS_CACHE = {}      # one clustering per training part, shared by both SMOTE settings


def _elbow(ks, inertias):
    """Elbow of the inertia curve: the k farthest from the straight line
    joining its first and last points, with log k on the x axis and both
    axes rescaled to [0, 1]."""
    x = np.log(np.asarray(ks, dtype=float))
    y = np.asarray(inertias, dtype=float)
    x = (x - x[0]) / (x[-1] - x[0])
    y = (y - y[-1]) / (y[0] - y[-1])
    return ks[int(np.argmax(np.abs(x + y - 1.0)))]


def kmeans_groups(X_tr, cache_key=None):
    """({cluster name: [genes]}, k) from k-means on the genes' standardised
    expression profiles over the training samples. No labels are used; k is
    the elbow of the within-cluster sum of squares over KMEANS_K_GRID."""
    if cache_key is not None and cache_key in _KMEANS_CACHE:
        return _KMEANS_CACHE[cache_key]
    profiles = X_tr.T.to_numpy(dtype=float)                 # genes x training samples
    grid = [k for k in KMEANS_K_GRID if k < profiles.shape[0]]
    fits = [KMeans(n_clusters=k, n_init=1, max_iter=100, random_state=RANDOM_STATE).fit(profiles)
            for k in grid]
    k = _elbow(grid, [f.inertia_ for f in fits]) if len(grid) > 2 else grid[-1]
    labels = fits[grid.index(k)].labels_
    genes = np.asarray(X_tr.columns)
    groups = {'cluster_%d' % c: list(genes[labels == c]) for c in np.unique(labels)}
    if cache_key is not None:
        _KMEANS_CACHE[cache_key] = (groups, k)
    return groups, k


# =========================================================================
# (4) One outer fold, one method
# =========================================================================
def run_one_outer_fold(X, y, train_idx, test_idx, method, group_cols, smote_flag, study=None):
    """Everything below happens on the training fold; the test fold is used once."""
    X_tr_raw, X_te_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    if study is None:
        X_tr, X_te = _impute_and_scale(X_tr_raw, X_te_raw)
    else:
        X_tr, X_te = _impute_and_scale(X_tr_raw, X_te_raw, study[train_idx], study[test_idx])

    selected_group = None

    if method in BI_TAXONOMIES:
        # ---- Eq. (3) ranking and Eq. (4) selection, inside this fold ----
        phi = rank_groups_phi(X_tr, y_tr, group_cols)
        if not phi:
            return None
        selected_group = max(phi, key=phi.get)
        cols = group_cols[selected_group]
        X_tr_sel, X_te_sel = X_tr[cols], X_te[cols]

    elif method == KMEANS:
        # the same Eq. (3)-(4) selection, on clusters learned from this training part
        groups, k = kmeans_groups(X_tr, cache_key=(id(X), X.shape, train_idx.tobytes()))
        phi = rank_groups_phi(X_tr, y_tr, groups)
        if not phi:
            return None
        cols = groups[max(phi, key=phi.get)]
        X_tr_sel, X_te_sel = X_tr[cols], X_te[cols]
        selected_group = 'k=%d' % k             # cluster labels do not correspond across folds

    elif method == 'pca':
        n_comp = select_baseline_config_inner(X_tr, y_tr, 'pca', smote_flag)
        pca = PCA(n_components=n_comp).fit(X_tr)
        cols = ['PC_%d' % i for i in range(pca.n_components_)]
        X_tr_sel = pd.DataFrame(pca.transform(X_tr), index=X_tr.index, columns=cols)
        X_te_sel = pd.DataFrame(pca.transform(X_te), index=X_te.index, columns=cols)
        selected_group = 'n_components=%s' % n_comp

    else:  # mrmr / lasso
        k = select_baseline_config_inner(X_tr, y_tr, method, smote_flag)
        ranked = (_rank_features_mrmr(X_tr, y_tr, k) if method == 'mrmr'
                  else _rank_features_lasso(X_tr, y_tr, k))
        cols = ranked[:k]
        if not cols:
            return None
        X_tr_sel, X_te_sel = X_tr[cols], X_te[cols]
        selected_group = '%s_top%d' % (method, k)

    # ---- classifier chosen by the inner CV of this same fold ----
    name, model, inner_score = select_classifier_inner(X_tr_sel, y_tr, smote_flag)
    if model is None:
        return None

    # ---- the outer test fold is touched exactly here, once ----
    y_pred = model.predict(X_te_sel)
    y_prob = model.predict_proba(X_te_sel)

    return {
        'y_true': np.asarray(y_te),
        'y_pred': np.asarray(y_pred),
        'y_prob': np.asarray(y_prob),
        'study': None if study is None else np.asarray(study[test_idx]),
        'group': selected_group,
        'model': name,
        'n_features': X_te_sel.shape[1],
        'inner_score': inner_score,
        'phi_top': (sorted(phi.items(), key=lambda kv: -kv[1])[:5]
                    if method in BI_TAXONOMIES + (KMEANS,) else None),
    }


# =========================================================================
# (5) One cohort, one method, all outer folds
# =========================================================================
def run_method(X, y, method, X_groups, smote_flag, verbose=True, study=None):
    group_cols = (_group_columns(X_groups[method], X.columns)
                  if method in BI_TAXONOMIES else None)

    # a pooled cohort is stratified on study and response together, so every
    # study is represented in every training fold
    strata = y if study is None else np.char.add(study.astype(str), y.astype(str))

    folds = []
    for fold_num, (tr, te) in enumerate(_outer_cv().split(X.values, strata), start=1):
        if verbose:
            print('    fold %d/%d ...' % (fold_num, N_OUTER_FOLDS), flush=True)
        out = run_one_outer_fold(X, y, tr, te, method, group_cols, smote_flag, study)
        if out is None:
            print('    fold %d produced no result and was skipped' % fold_num, flush=True)
            continue
        out['fold'] = fold_num
        folds.append(out)

    if not folds:
        return None

    oof_true = np.concatenate([f['y_true'] for f in folds])
    oof_pred = np.concatenate([f['y_pred'] for f in folds])
    oof_prob = np.concatenate([f['y_prob'] for f in folds])

    group_counts = Counter(f['group'] for f in folds)
    modal_group, modal_count = group_counts.most_common(1)[0]
    model_counts = Counter(f['model'] for f in folds)

    brier, cal_int, cal_slope = utils.get_calibration_metrics(oof_true, oof_prob[:, 1])

    return {
        'method': method,
        'smote': smote_flag,
        'oof_y_true': oof_true,
        'oof_y_pred': oof_pred,
        'oof_y_prob': oof_prob,
        'oof_study': (None if study is None
                      else np.concatenate([f['study'] for f in folds])),
        'fold_group': {f['fold']: f['group'] for f in folds},
        'fold_model': {f['fold']: f['model'] for f in folds},
        'fold_n_features': {f['fold']: f['n_features'] for f in folds},
        'fold_phi_top': {f['fold']: f['phi_top'] for f in folds},
        'selected_group': modal_group,              # Table VII, 'Group'
        'selection_count': int(modal_count),        # Table VII, 'Sel.'  (out of 10)
        'group_counts': dict(group_counts),
        'best_model': model_counts.most_common(1)[0][0],
        'model_counts': dict(model_counts),
        'mean_n_features': float(np.mean([f['n_features'] for f in folds])),
        'acc': accuracy_score(oof_true, oof_pred),
        'acc_ci': utils.get_bootstrap_ci(oof_true, oof_pred, accuracy_score, n_jobs=-1),
        'balanced_acc': balanced_accuracy_score(oof_true, oof_pred),
        'AUC': roc_auc_score(oof_true, oof_prob[:, 1]),
        'AUC_ci': utils.get_bootstrap_ci(oof_true, oof_prob[:, 1], roc_auc_score, n_jobs=-1),
        'PR_AUC': average_precision_score(oof_true, oof_prob[:, 1]),
        'PR_AUC_ci': utils.get_bootstrap_ci(oof_true, oof_prob[:, 1],
                                            average_precision_score, n_jobs=-1),
        'brier_score': brier,
        'calib_intercept': cal_int,
        'calib_slope': cal_slope,
        'n_folds_used': len(folds),
    }


def run_cohort(cohort_name, X, y, X_groups, out_dir, smote_flags=(0, 1),
               methods=ALL_METHODS, study=None, merge_existing=False):
    """Run every method for one cohort and save one pickle per smote setting.

    `study` (one entry per sample) marks a cohort pooled from several studies:
    genes are then standardised within each study and the outer folds are
    stratified on study and response.

    With merge_existing, the methods run here are added to the cohort's
    existing pickles instead of replacing them (e.g. to add k-means to a
    finished run). The paired tests need every method scored on the same
    outer folds, so the new predictions must line up sample-for-sample with
    the existing ones; otherwise nothing is saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    _KMEANS_CACHE.clear()
    y = np.asarray(y).ravel()
    if study is not None:
        study = np.asarray(study).ravel()
        if len(study) != len(y):
            raise ValueError('%s: %d study labels for %d samples' % (cohort_name, len(study), len(y)))
    all_results = {}

    for smote_flag in smote_flags:
        results = {}
        for method in methods:
            print('[%s] smote=%d  method=%s' % (cohort_name, smote_flag, method), flush=True)
            res = run_method(X, y, method, X_groups, smote_flag, study=study)
            if res is not None:
                results[method] = res
                print('    -> AUC %.4f | group %r selected in %d/%d folds | model %s'
                      % (res['AUC'], res['selected_group'], res['selection_count'],
                         res['n_folds_used'], res['best_model']), flush=True)

        out_file = os.path.join(out_dir, 'corrected_%s_smote%d.pkl' % (cohort_name, smote_flag))
        if merge_existing and os.path.exists(out_file):
            with open(out_file, 'rb') as f:
                existing = pickle.load(f)
            for method, res in results.items():
                for kept in (m for m in existing if m not in results):
                    # compare positives only, so older pickles with non-0/1 labels still match
                    if not np.array_equal(np.asarray(existing[kept]['oof_y_true']) == 1,
                                          np.asarray(res['oof_y_true']) == 1):
                        raise ValueError('%s smote=%d: %s predictions do not line up with the existing '
                                         '%s results; the data or folds differ, nothing saved'
                                         % (cohort_name, smote_flag, method, kept))
            existing.update(results)
            results = existing
        with open(out_file, 'wb') as f:
            pickle.dump(results, f)
        print('saved %s' % out_file, flush=True)
        all_results[smote_flag] = results

    return all_results
