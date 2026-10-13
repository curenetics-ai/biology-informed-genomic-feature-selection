import os
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from utils import utils

###################################################################################################################
########################################## Load gene expression data ##############################################
###################################################################################################################
path_lusc = 'C:/Users/farid/Downloads/bi paper/kmeans vs biology informed feature selection/luad relapse prediction'
path_data = os.path.join(path_lusc, 'Data/luad-rsem-fpkm-tcga-t.txt.gz')

data_cancer = pd.read_csv(path_data, sep='\t')
data_cancer.index = data_cancer.Hugo_Symbol
all_data_cancer = data_cancer.drop(columns=['Hugo_Symbol', 'Entrez_Gene_Id']).T

all_data_cancer = all_data_cancer.dropna(axis=1)

all_data_cancer.index = [i[:15] for i in all_data_cancer.index]

###################################################################################################################
############################################## Load clinical data #################################################
###################################################################################################################

clinical_data = pd.read_csv(os.path.join(path_lusc, 'Data/clinical_survival.csv'), sep='\t', index_col=0)

clinical_data_recurrence = clinical_data[['DFI', 'DFI.time']]
clinical_data_recurrence['DFI'] = clinical_data_recurrence['DFI'].astype(bool)
clinical_data_recurrence = clinical_data_recurrence.dropna()

all_data_cancer = utils.median_gene_data(all_data_cancer.T).sort_index().T
common_samples = list(set(clinical_data_recurrence.index) & set(all_data_cancer.index))

X = all_data_cancer.loc[common_samples].sort_index()
clinical_data_recurrence = clinical_data_recurrence.loc[common_samples].sort_index()

print(list(X.index) == list(clinical_data_recurrence.index))
y_within_5_years = pd.DataFrame(np.array([0 if i > 365 * 5 else 1 for i in clinical_data_recurrence['DFI.time']]),
                                columns=['labels'], index=X.index)

seq_data = pd.read_excel(os.path.join(path_lusc, 'Data/nucletoide_sequences.xlsx'), index_col = 0, engine='openpyxl')

clinical_data_recurrence['DFI'] = y_within_5_years
y = pd.DataFrame(y_within_5_years, index = clinical_data_recurrence.index)

seq_data = pd.read_excel(os.path.join(path_lusc, 'Data/nucletoide_sequences.xlsx'), index_col = 0, engine='openpyxl')

with open(os.path.join(path_lusc, 'Data/pathway_lists.pkl'), 'rb') as f:
    pathway_list = pickle.load(f)
results_all = {}


common_genes = list(set(X.columns) & set(seq_data.hgnc_symbol))
X = np.log2(X[common_genes].reindex(sorted(X[common_genes].columns), axis=1)+1)







# In[ ]:


from scipy.special import logit
for smote_flag in [0, 1]:
    results_all = {}
    for fs_method in ['chromosome_based', 'region_based', 'pathway_based', 'lasso', 'mrmr', 'pca']:
        if fs_method == 'region_based':
            seq_data_1 = seq_data[(seq_data['chromosome_name'] == 7) & (seq_data['band'] == 'q11.23')]
            X_1 = utils.region_based_fs(X, seq_data_1)
        if fs_method == 'chromosome_based':
            seq_data_1 = seq_data[(seq_data['chromosome_name'] == 7)]
            X_1 = utils.chromosome_based_fs(X, seq_data_1)
        if fs_method == 'pathway_based':
            pathway_list_1 = 'REACTOME_DOWNREGULATION_OF_TGF_BETA_RECEPTOR_SIGNALING'
            X_1 = utils.pathway_based_fs(X, {pathway_list_1 : pathway_list[pathway_list_1]})
        if 'based' in fs_method:
            for feat_name in list(X_1.keys()):
                results_all[fs_method+str(feat_name)] = utils.traditional_ml_kfold(X_1[feat_name], y, path_lusc,
                                                              smote_flag, feat_name, 10, fs_method)


        if fs_method == 'lasso' or fs_method == 'mrmr':
            for feat_name in [50, 100, 200, 500, 1000, 2000, 5000]:
                results_all[fs_method+str(feat_name)] = utils.traditional_ml_kfold(X, y, path_lusc,
                                                              smote_flag, feat_name, 10, fs_method)

        if fs_method == 'pca':
            for feat_name in [0.95]:
                results_all[fs_method+str(feat_name)] = utils.traditional_ml_kfold(X, y, path_lusc,
                                                             smote_flag, feat_name, 10, fs_method)


    # In[ ]:
    if smote_flag == 0:
        with open('results_luad_no_smote_all.pkl', 'wb') as f:
            pickle.dump(results_all, f)
    else:
        with open('results_luad_smote_all.pkl', 'wb') as f:
            pickle.dump(results_all, f)

    utils.export_results_to_excel(results_all, smote_flag, out_dir=path_lusc)

