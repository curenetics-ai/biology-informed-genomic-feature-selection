"""
run_corrected.py
================
Driver for the corrected nested cross-validation (Reviewer 3, comment R3.1).

Loads each cohort from the same files and with the same normalisation as the
existing per-cohort scripts, then runs the corrected protocol from
corrected_nested_cv.py and writes the tables with build_tables.py.

LABELS
------
Two fixes relative to the submitted scripts:

  * Response labels (kidney, stomach) are mapped with a check: a value that is
    not 'R', 'NR' or 'N' stops the run and is printed, instead of silently
    becoming NaN. Before the cohort is run, every cohort's labels must be
    exactly 0/1; samples with no label are dropped and counted.

  * Relapse labels (LUAD, LUSC) now use the DFI event flag. Previously every
    patient with DFI.time <= 5 years was labelled 'relapse', including patients
    censored without a recorded relapse. See DROP_CENSORED_BEFORE_HORIZON.

KIDNEY COHORT
-------------
kidney_data.csv pooled four studies stored on different scales, one of them
(Mariathasan, IMvigor210) urothelial carcinoma. The kidney cohort is now the
Braun et al. (2020) ccRCC study alone, read from ICB_Braun_data.csv (unlogged
TPM) and analysed as log2(TPM + 1). Its Ensembl IDs are mapped to gene symbols
with mygene.info (pip install mygene); the first run saves the mapping to
kidney/Data/braun_ensembl_to_symbol_mygene.csv and later runs reuse it.

MELANOMA COHORT
---------------
melanoma_data.csv was a pre-standardised merge of four studies whose
processing could not be reconstructed. The melanoma cohort is now pooled from
the raw TIGER files of four RNA-seq studies -- GSE78220 (Hugo 2016), GSE91061
(Riaz 2017), PRJEB23709 (Gide 2019) and phs000452 (Liu 2019) -- keeping one
pre-treatment sample per patient and the therapies in MELANOMA_THERAPIES.
Responses follow TIGER's response_NR (R = CR/PR/MR, N = SD/PD). Each study's
clinical table and expression file (Melanoma-<study>.GeneExpr.tsv) are found
under the melanoma Data folder, and whatever scale a study was stored on is
converted to log2(x + 1); corrected_nested_cv then standardises genes within
each study inside every fold and stratifies the folds on study and response.

HOW TO RUN
----------
  1. Put this file, corrected_nested_cv.py and build_tables.py next to your
     existing scripts, so that `from utils import utils` still resolves.
  2. Edit BASE below (and any cohort path that differs on your machine).
  3. Run one cohort at a time:

         python run_corrected.py kidney
         python run_corrected.py melanoma
         python run_corrected.py stomach
         python run_corrected.py luad
         python run_corrected.py lusc
         python run_corrected.py nsclc

     or all six in sequence:

         python run_corrected.py all

  4. To add the k-means counterpart to cohorts that already finished, without
     rerunning the other methods (results are merged into the existing pickles):

         python run_corrected.py kmeans            (all six cohorts)
         python run_corrected.py kmeans kidney     (one or more cohorts)

  5. When every cohort is finished, build the tables and the figure:

         python run_corrected.py tables

Results are written to  <BASE>/corrected_results/  and the manuscript-ready
tables to  <BASE>/manuscript_tables/ .

Each cohort is independent, so they can be run on different machines and the
pickles simply copied into the same corrected_results folder.
"""

import os
import sys
import glob
import pickle

import numpy as np
import pandas as pd

from utils import utils
import corrected_nested_cv as cnc
import build_tables


# =========================================================================
# EDIT THIS: the folder that holds the per-cohort data directories
# =========================================================================
BASE = '/home/hercule3/Documents/Ferid/datasets/'

COHORT_PATHS = {
    'kidney':   os.path.join(BASE, 'kidney'),
    'melanoma': os.path.join(BASE, 'melanoma immunotherapy response prediction'),
    'stomach':  os.path.join(BASE, 'stomach cancer immunotherapy response prediction'),
    'luad':     os.path.join(BASE, 'luad relapse prediction'),
    'lusc':     os.path.join(BASE, 'lusc relapse prediction'),
    'nsclc':    os.path.join(BASE, 'TCIA'),
}

RESULTS_DIR = os.path.join(BASE, 'corrected_results')
TABLES_DIR = os.path.join(BASE, 'manuscript_tables')


# =========================================================================
# Label definitions
# =========================================================================
# Immunotherapy response: responder = 1, non-responder = 0. 'N' is accepted
# because some clinical files code non-responders that way.
RESPONSE_MAP = {'R': 1, 'NR': 0, 'N': 0}

# Relapse cohorts (LUAD, LUSC): label 1 = relapse within this many days.
RELAPSE_HORIZON_DAYS = 365 * 5

# True : patients censored (DFI == 0) before the horizon are dropped, because
#        whether they relapse within 5 years is unknown. Standard definition.
# False: reproduces the submitted analysis, where every patient with
#        DFI.time <= 5 years was labelled 1, whether or not a relapse was
#        recorded.
DROP_CENSORED_BEFORE_HORIZON = True

# Melanoma: the TIGER studies pooled into one cohort. Each study needs its
# clinical table and its expression file somewhere under MELANOMA_DIR (sub-
# folders included), named Melanoma-<study>.<anything>.tsv; the two are told
# apart by content, and Melanoma-<study>.GeneExpr.tsv is preferred when a
# study has more than one expression file.
MELANOMA_STUDIES = ('GSE78220', 'GSE91061', 'PRJEB23709', 'phs000452')
MELANOMA_DIR = os.path.join(COHORT_PATHS['melanoma'], 'Data')
# anti-PD-1 monotherapy only (237 patients); add 'anti-CTLA-4+anti-PD-1' to
# include Gide's 32 combination-therapy patients (269 in total)
MELANOMA_THERAPIES = ('anti-PD-1',)


def _map_response(raw, cohort):
    """'R' -> 1, 'NR' / 'N' -> 0. A missing label stays NaN and is dropped in
    _check_labels; any other value stops the run."""
    text = raw.astype('string').str.strip()
    y = text.map(RESPONSE_MAP)
    unrecognised = text[text.notna() & y.isna()]
    if len(unrecognised):
        raise ValueError('%s: response values not recognised: %s (expected one of %s)'
                         % (cohort, unrecognised.value_counts().to_dict(), sorted(RESPONSE_MAP)))
    return y.astype('float')


def _relapse_labels(clinical, cohort):
    """Relapse within RELAPSE_HORIZON_DAYS from the TCGA disease-free interval.

    DFI = 1: new tumour event at DFI.time.  DFI = 0: censored at DFI.time.
    Returns a 0/1 Series indexed like `clinical`, without the dropped patients.
    """
    if not DROP_CENSORED_BEFORE_HORIZON:
        t = clinical['DFI.time'].dropna()
        y = (t <= RELAPSE_HORIZON_DAYS).astype(int)
        print('  %s: %d labelled relapse, %d relapse-free (DFI event flag ignored)'
              % (cohort, int(y.sum()), int((y == 0).sum())))
        return y

    rec = clinical[['DFI', 'DFI.time']].dropna()
    flags = set(rec['DFI'].unique().tolist())
    if not flags <= {0, 1}:
        raise ValueError('%s: DFI must be 0/1, got %s' % (cohort, sorted(flags, key=str)))

    event = rec['DFI'] == 1
    within = rec['DFI.time'] <= RELAPSE_HORIZON_DAYS
    unknown = within & ~event
    print('  %s: %d relapses within the horizon, %d relapse-free past it, '
          '%d censored before it (dropped)'
          % (cohort, int((within & event).sum()), int((~within).sum()), int(unknown.sum())))
    return (within & event)[~unknown].astype(int)


def _check_labels(cohort, X, y, study=None):
    """Drop samples with no label and require the rest to be exactly 0/1.

    Only the rows of X (and of `study`, for a pooled cohort) are filtered:
    corrected_nested_cv uses the taxonomy groups for their column names alone,
    so they need no change.
    """
    y = pd.Series(np.asarray(y).ravel())
    if len(y) != len(X):
        raise ValueError('%s: %d labels for %d samples' % (cohort, len(y), len(X)))
    print('  labels as loaded: %s' % y.value_counts(dropna=False).to_dict())

    missing = y.isna().to_numpy()
    if missing.any():
        print('  dropping %d samples with no label' % int(missing.sum()))
        X, y = X.iloc[~missing], y[~missing]
        if study is not None:
            study = np.asarray(study)[~missing]

    values = set(y.unique().tolist())
    if values != {0, 1}:
        raise ValueError('%s: labels must be 0 and 1, got %s; fix the loader'
                         % (cohort, sorted(values, key=str)))
    return X, y.astype(int).to_numpy(), study


# =========================================================================
# Shared tail: intersect with the annotation and build the three taxonomies
# =========================================================================
def _finalise(X, seq_data, pathway_list, log2=False):
    common_genes = list(set(X.columns) & set(seq_data.hgnc_symbol))
    X = X[common_genes].reindex(sorted(X[common_genes].columns), axis=1)
    if log2:
        X = np.log2(X + 1)

    X_groups = {
        'pathway_based': utils.pathway_based_fs(X, pathway_list),
        'region_based': utils.region_based_fs(X, seq_data),
        'chromosome_based': utils.chromosome_based_fs(X, seq_data),
    }
    for k, v in X_groups.items():
        print('  %-18s %d groups' % (k, len(v)))
    return X, X_groups


def _load_annotation(path, sub='Data'):
    seq = pd.read_excel(os.path.join(path, sub, 'nucletoide_sequences.xlsx'),
                        index_col=0, engine='openpyxl')
    with open(os.path.join(path, sub, 'pathway_lists.pkl'), 'rb') as f:
        pathways = pickle.load(f)
    return seq, pathways


# =========================================================================
# Per-cohort loaders -- same files and normalisation as the existing scripts
# =========================================================================
def _ensembl_to_symbol(ids, cache_path):
    """Ensembl gene ID (version removed) -> gene symbol, from mygene.info.

    mygene.info is updated regularly, so the first run saves the mapping to
    cache_path and every later run reuses it: reruns stay identical and the
    file documents the mapping used. Delete the file to query again.
    """
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0)['symbol']

    import mygene
    mg = mygene.MyGeneInfo()
    res = mg.querymany(sorted(set(ids)), scopes='ensembl.gene', fields='symbol',
                       species='human', as_dataframe=True, verbose=False)
    if 'notfound' in res.columns:
        res = res[res['notfound'] != True]
    res = res[res['symbol'].notna()]
    if '_score' in res.columns:                       # several hits for one ID: keep the best
        res = res.sort_values('_score', ascending=False, kind='mergesort')
    symbol = res.loc[~res.index.duplicated(keep='first'), 'symbol']
    symbol.index.name = 'ensembl_gene_id'
    symbol.to_frame().to_csv(cache_path)
    print('  mygene.info: %d of %d Ensembl IDs mapped; saved to %s'
          % (len(symbol), len(set(ids)), cache_path))
    return symbol


def load_kidney():
    """ccRCC treated with anti-PD-1: the Braun et al. (2020) cohort, read from its
    own file. kidney_data.csv pooled four studies on different scales, one of
    them (Mariathasan, IMvigor210) urothelial carcinoma, and reused sample names
    across studies. Braun's file holds unlogged TPM with Ensembl IDs as rows."""
    p = COHORT_PATHS['kidney']
    tpm = pd.read_csv(os.path.join(p, 'Data/ICB_Braun_data.csv'), index_col=0)
    tpm = tpm[~tpm.index.astype(str).str.endswith('_PAR_Y')]        # chrY copies of PAR genes
    tpm.index = tpm.index.astype(str).str.split('.').str[0]
    X = tpm.T.astype(float)                                           # samples x Ensembl IDs
    if (X.max(axis=1) <= 25).any():
        raise ValueError('kidney: some Braun samples look log-transformed already')

    symbol = _ensembl_to_symbol(X.columns, os.path.join(p, 'Data', 'braun_ensembl_to_symbol_mygene.csv'))
    X = X.loc[:, X.columns.isin(symbol.index)]
    X.columns = symbol.loc[X.columns].to_numpy()
    X = utils.median_gene_data(X)                                     # IDs sharing a symbol: median

    lab = pd.read_csv(os.path.join(p, 'Data/ICB_Braun_labels.csv')).set_index('case_id')
    lab = lab.loc[~lab.index.duplicated(keep='first')]
    common = sorted(set(X.index) & set(lab.index))
    X = X.loc[common]
    print('  kidney (Braun): %d samples, %d genes with a symbol' % X.shape)
    y = _map_response(lab.loc[common, 'response'], 'kidney')

    # the annotation is still needed for chromosome and band; genes whose
    # mygene symbol is absent from it are dropped by _finalise
    seq, pw = _load_annotation(p)
    X, groups = _finalise(X, seq, pw, log2=True)                      # log2(TPM + 1)
    print('  kidney (Braun): %d genes also in the annotation' % X.shape[1])
    return X, y.values, groups


def _tiger_files(study):
    """(clinical table, expression file) of one TIGER study under MELANOMA_DIR.
    The clinical table is the file with a response_NR column."""
    clinical, expression = [], []
    for f in sorted(glob.glob(os.path.join(MELANOMA_DIR, '**', 'Melanoma-%s.*tsv' % study), recursive=True)):
        head = pd.read_csv(f, sep='\t', nrows=3)
        (clinical if 'response_NR' in head.columns else expression).append(f)
    expression.sort(key=lambda f: 'GeneExpr' not in os.path.basename(f))
    if not clinical or not expression:
        raise ValueError('%s: need a clinical table and an expression file under %s; found %s'
                         % (study, MELANOMA_DIR, clinical + expression))
    print('  %-11s clinical: %s | expression: %s'
          % (study, os.path.relpath(clinical[0], MELANOMA_DIR), os.path.relpath(expression[0], MELANOMA_DIR)))
    return clinical[0], expression[0]


def _melanoma_samples(study, path):
    """One pre-treatment sample per patient with a known response, from TIGER's
    clinical table, indexed by sample_id."""
    c = pd.read_csv(path, sep='\t')
    # phs000452 leaves Treatment blank; its biopsies are pre-treatment
    pre = c['Treatment'].fillna('PRE').astype(str).str.upper() == 'PRE'
    c = c[pre & c['response_NR'].isin(['R', 'N']) & c['Therapy'].isin(MELANOMA_THERAPIES)]

    name = c['patient_name'].astype(str)
    if study == 'phs000452':
        # 20 patients also appear as 'IPI_punch_tumor_PatNN' with identical
        # outcomes: older (pre-ipilimumab) biopsies of the same patients. Keep
        # Liu et al.'s pre-anti-PD-1 'PatientNN_*' samples only.
        c, name = c[name.str.startswith('Patient')], name[name.str.startswith('Patient')]
        patient = name.str.extract(r'^(Patient\d+)', expand=False)
    elif study == 'GSE78220':
        patient = name.str.replace(r'(\d+)[A-Z]$', r'\1', regex=True)    # Pt27A / Pt27B
    else:
        patient = name
    c = c.assign(patient=patient.to_numpy()).drop_duplicates('patient', keep='first')
    return c.set_index('sample_id')


def _to_log2p1(X, study):
    """Put one study's samples x genes matrix on log2(x + 1), whichever of the
    three usual scales it was stored on. Prints what it found."""
    sample_max = X.max(axis=1)
    if (sample_max > 25).all():                    # no log2 expression value reaches 25
        print('  %-11s stored unlogged: log2(x + 1) applied' % study)
        return np.log2(X.clip(lower=0) + 1)
    if (sample_max > 25).any():
        raise ValueError('%s: some samples are logged and others are not' % study)
    floor = float(np.nanmin(X.to_numpy()))
    if floor >= 0:
        print('  %-11s stored as log2(x + 1): used as is' % study)
        return X
    pseudo = 2.0 ** floor                          # log2(x + p) stores a zero as log2(p)
    print('  %-11s stored as log2(x + %.3g): converted to log2(x + 1)' % (study, pseudo))
    return np.log2(np.clip(2.0 ** X - pseudo, 0, None) + 1)


def _tiger_expression(study, clin, path):
    """Samples x gene symbols on log2(x + 1) for the samples in `clin`.

    The file's columns may be run IDs or patient names, and its genes symbols
    or Ensembl IDs (mapped with mygene, cached like the kidney mapping). Rows
    sharing a symbol are combined by their median.
    """
    E = pd.read_csv(path, sep='\t')
    names = {'sample_id': clin.index.to_series().astype(str),
             'patient_name': clin['patient_name'].astype(str)}
    wanted = max(names.values(), key=lambda n: n.isin(E.columns).sum())
    missing = wanted[~wanted.isin(E.columns)]
    if len(missing):
        raise ValueError('%s: %d selected samples missing from %s, e.g. %s'
                         % (study, len(missing), os.path.basename(path), list(missing[:3])))
    # pandas renames a repeated column 'x' to 'x.1': a patient with several
    # samples (pre and on treatment) under one name cannot be told apart
    ambiguous = [w for w in wanted if '%s.1' % w in E.columns]
    if ambiguous:
        raise ValueError('%s: %s names samples by patient, and %d patients (e.g. %s) have more than '
                         'one sample, so the pre-treatment one cannot be identified'
                         % (study, os.path.basename(path), len(ambiguous), ambiguous[:3]))

    # gene identifiers: a text column (a symbol column first), otherwise the index
    text = [c for c in E.columns
            if c not in set(wanted) and not pd.api.types.is_numeric_dtype(E[c])]
    text.sort(key=lambda c: 0 if 'SYMBOL' in str(c).upper() else 1 if 'NAME' in str(c).upper() else 2)
    genes = (E[text[0]] if text else E.index.to_series()).astype(str).str.strip()
    keep = (genes != '') & (genes.str.lower() != 'nan')

    X = E.loc[keep.to_numpy(), wanted.tolist()].astype(float)
    X.index = genes[keep].to_numpy()
    X = X.T
    X.index = clin.index                                   # sample_id, in clin's order
    X = _to_log2p1(X, study)

    if pd.Index(X.columns).str.startswith('ENSG').mean() > 0.5:
        X.columns = pd.Index(X.columns).str.split('.').str[0]
        symbol = _ensembl_to_symbol(
            X.columns, os.path.join(MELANOMA_DIR, 'melanoma_%s_ensembl_to_symbol_mygene.csv' % study))
        X = X.loc[:, X.columns.isin(symbol.index)]
        X.columns = symbol.loc[X.columns].to_numpy()
    return utils.median_gene_data(X)


def load_melanoma():
    """Pre-treatment melanoma cohort pooled from four TIGER studies; see the
    module docstring. Returns the study of each sample as a fourth value."""
    p = COHORT_PATHS['melanoma']
    parts, labels, study = [], [], []
    for s in MELANOMA_STUDIES:
        clinical_path, expression_path = _tiger_files(s)
        clin = _melanoma_samples(s, clinical_path)
        Xs = _tiger_expression(s, clin, expression_path)
        ys = clin.loc[Xs.index, 'response_NR'].map({'R': 1, 'N': 0})
        print('  %-11s %3d patients, %3d responders' % (s, len(Xs), int(ys.sum())))
        parts.append(Xs)
        labels.append(ys)
        study += [s] * len(Xs)

    genes = sorted(set.intersection(*[set(x.columns) for x in parts]))
    X = pd.concat([x[genes] for x in parts])
    y = pd.concat(labels)
    seq, pw = _load_annotation(p)
    X, groups = _finalise(X, seq, pw)                     # already log2(x + 1)
    return X, y.values, groups, np.array(study)


def load_stomach():
    p = COHORT_PATHS['stomach']
    X = pd.read_csv(os.path.join(p, 'Data/STAD-PRJEB25780.Response.tsv'),
                    index_col=1, sep='\t').drop(columns='GENE_SYMBOL').T
    y = pd.DataFrame(pd.read_csv(os.path.join(p, 'Data/STAD-PRJEB25780.Response_clinical.tsv'),
                                 index_col=1, sep='\t')['response_NR'])
    y['response_NR'] = _map_response(y['response_NR'], 'stomach')
    seq, pw = _load_annotation(p)
    X, groups = _finalise(X, seq, pw)
    y = y.loc[X.index]
    return X, y['response_NR'].values, groups


def load_nsclc():
    p = COHORT_PATHS['nsclc']
    X = pd.read_csv(os.path.join(p, 'GSE103584_R01_NSCLC_RNAseq.txt'),
                    sep='\t', index_col=0).dropna().T
    X.index = ['.'.join(i) for i in X.index.str.split('-')]
    y = pd.read_csv(os.path.join(p, 'Clinical_info_filtered.csv'), index_col=0)
    X = X.loc[y.index]
    seq, pw = _load_annotation(p, sub='')
    X, groups = _finalise(X, seq, pw)
    return X, y['Recurrence'].values, groups


def _load_tcga_relapse(cohort, expr_file, clinical_file):
    p = COHORT_PATHS[cohort]
    data_cancer = pd.read_csv(os.path.join(p, 'Data', expr_file), sep='\t')
    data_cancer.index = data_cancer.Hugo_Symbol
    all_data = data_cancer.drop(columns=['Hugo_Symbol', 'Entrez_Gene_Id']).T
    all_data = all_data.dropna(axis=1)
    all_data.index = [i[:15] for i in all_data.index]

    clinical = pd.read_csv(os.path.join(p, 'Data', clinical_file), sep='\t', index_col=0)
    labels = _relapse_labels(clinical, cohort)

    all_data = utils.median_gene_data(all_data.T).sort_index().T
    common = sorted(set(labels.index) & set(all_data.index))
    X = all_data.loc[common]
    y = labels.loc[common].to_numpy()

    seq, pw = _load_annotation(p)
    X, groups = _finalise(X, seq, pw, log2=True)
    return X, y, groups


def load_luad():
    return _load_tcga_relapse('luad', 'luad-rsem-fpkm-tcga-t.txt.gz', 'clinical_survival.csv')


def load_lusc():
    return _load_tcga_relapse('lusc', 'lusc-rsem-fpkm-tcga-t.txt.gz', 'survival_LUSC_survival.txt')


LOADERS = {
    'kidney': load_kidney, 'melanoma': load_melanoma, 'stomach': load_stomach,
    'luad': load_luad, 'lusc': load_lusc, 'nsclc': load_nsclc,
}


# =========================================================================
def run(cohort, methods=cnc.ALL_METHODS, merge_existing=False):
    print('=' * 70)
    print('COHORT:', cohort, '' if methods == cnc.ALL_METHODS else '| methods: %s' % ', '.join(methods))
    print('=' * 70)
    loaded = LOADERS[cohort]()
    X, y, groups = loaded[:3]
    study = loaded[3] if len(loaded) > 3 else None          # pooled cohorts only
    X, y, study = _check_labels(cohort, X, y, study)
    print('  X: %d samples x %d genes | positives: %d / %d'
          % (X.shape[0], X.shape[1], int(y.sum()), len(y)))
    if study is not None:
        for s in np.unique(study):
            print('    %-11s %3d samples, %.0f%% responders' % (s, (study == s).sum(), 100 * y[study == s].mean()))
    cnc.run_cohort(cohort, X, y, groups, RESULTS_DIR, methods=methods, study=study,
                   merge_existing=merge_existing)


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    cohorts = ['kidney', 'melanoma', 'stomach', 'luad', 'lusc', 'nsclc']

    if arg == 'tables':
        build_tables.build_all(RESULTS_DIR, TABLES_DIR)
    elif arg == 'all':
        for c in cohorts:
            run(c)
        build_tables.build_all(RESULTS_DIR, TABLES_DIR)
    elif arg == 'kmeans':
        # add only the k-means counterpart to finished runs
        for c in sys.argv[2:] or cohorts:
            run(c, methods=(cnc.KMEANS,), merge_existing=True)
    elif arg in LOADERS:
        run(arg)
    else:
        print(__doc__)
        print('unknown argument %r; expected one of: %s, all, kmeans, tables'
              % (arg, ', '.join(LOADERS)))
