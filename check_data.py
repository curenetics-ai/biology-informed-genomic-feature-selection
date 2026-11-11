import os, numpy as np, pandas as pd
import run_corrected as rc
import run_corrected as rc
import os, pandas as pd
import run_corrected as rc
p = rc.COHORT_PATHS['luad']
header = pd.read_csv(os.path.join(p, 'Data', 'luad-rsem-fpkm-tcga-t.txt.gz'), sep='\t', nrows=0)
samples = {c[:15] for c in header.columns if c not in ('Hugo_Symbol', 'Entrez_Gene_Id')}
clin = pd.read_csv(os.path.join(p, 'Data', 'clinical_survival.csv'), sep='\t', index_col=0)
clin = clin.loc[clin.index.isin(samples)]
rec = clin[['DFI', 'DFI.time']].dropna()
within = rec['DFI.time'] <= 365 * 5
print('LUAD patients with RNA-seq: %d | with DFI data: %d' % (len(clin), len(rec)))
print('relapse within 5 years: %d | relapse-free beyond 5 years: %d | censored before 5 years (excluded): %d'
      % ((within & (rec.DFI == 1)).sum(), (~within).sum(), (within & (rec.DFI == 0)).sum()))
'''for c in ['stomach', 'luad', 'lusc', 'nsclc']:
    X, y, g = rc.LOADERS[c]()[:3]
    X, y, _ = rc._check_labels(c, X, y)
    print(c, 'patients x genes:', X.shape, '| positives:', int(y.sum()))
for cohort in ['kidney', 'melanoma']:
    loaded = rc.LOADERS[cohort]()
    X, y, groups = loaded[:3]
    study = loaded[3] if len(loaded) > 3 else None
    X, y, study = rc._check_labels(cohort, X, y, study)
    print('\n%s: %d patients x %d genes | %d responders (%.1f%%)'
          % (cohort.upper(), X.shape[0], X.shape[1], y.sum(), 100 * y.mean()))
    for name, g in groups.items():
        sizes = [d.shape[1] for d in g.values()]
        print('  %-17s %4d groups, %d to %d genes each (median %d)'
              % (name, len(g), min(sizes), max(sizes), int(np.median(sizes))))
    if study is not None:
        for s in np.unique(study):
            print('  %-11s %3d patients, %3d responders' % (s, (study == s).sum(), y[study == s].sum()))
    if cohort == 'kidney':
        lab = pd.read_csv(os.path.join(rc.COHORT_PATHS['kidney'], 'Data/ICB_Braun_labels.csv')).set_index('case_id')
        print('  treatment:', lab.loc[X.index, 'treatment'].value_counts(dropna=False).to_dict())'''