import os
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from utils import utils

###################################################################################################################
########################################## Load gene expression data ##############################################
###################################################################################################################

path_kidney = 'C:/Users/farid/Downloads/bi paper/kmeans vs biology informed feature selection/stomach cancer immunotherapy response prediction'

X = pd.read_csv(os.path.join(path_kidney, 'Data/STAD-PRJEB25780.Response.tsv'), index_col=1, sep='\t').drop(columns='GENE_SYMBOL').T
y = pd.DataFrame(pd.read_csv(os.path.join(path_kidney, 'Data/STAD-PRJEB25780.Response_clinical.tsv'), index_col=1, sep='\t')['response_NR'])
binary_mapping = {
    'N': 0,
    'R': 1
}

# 2. Map the dictionary to your column
y['response_NR'] = y['response_NR'].map(binary_mapping)

seq_data = pd.read_excel(os.path.join(path_kidney, 'Data/nucletoide_sequences.xlsx'), index_col = 0, engine='openpyxl')

with open(os.path.join(path_kidney, 'Data/pathway_lists.pkl'), 'rb') as f:
    pathway_list = pickle.load(f)
results_all = {}


common_genes = list(set(X.columns) & set(seq_data.hgnc_symbol))
X = X[common_genes].reindex(sorted(X[common_genes].columns), axis=1)







X_groups = {
    'pathway_based':    utils.pathway_based_fs(X, pathway_list),       # {pathway name: DataFrame}
    'region_based':     utils.region_based_fs(X, seq_data),            # {'band q23.3, chr num 1': DataFrame}
    'chromosome_based': utils.chromosome_based_fs(X, seq_data),        # {'chr num 12': DataFrame}
}
for k, v in X_groups.items():
    print(k, len(v), 'groups')          # sanity check: should be ~1600 pathways, not 4771

for smote_flag in [0, 1]:
    results_all = {}
    for fs_method in ['chromosome_based', 'region_based', 'pathway_based', 'lasso', 'mrmr', 'pca']:
        if fs_method in X_groups:
            res = utils.traditional_ml_kfold_many(X_groups[fs_method], y['response_NR'],
                                                  path_kidney, smote_flag, 10, fs_method)
            for feat_name, compiled in res.items():
                results_all[fs_method + str(feat_name)] = compiled
        # lasso / mrmr / pca: keep your existing traditional_ml_kfold(X, ...) calls
    utils.export_results_to_excel(results_all, smote_flag, out_dir=path_kidney)

from scipy.special import logit
'''for smote_flag in [0, 1]:
    results_all = {}
    for fs_method in ['chromosome_based', 'region_based', 'pathway_based', 'lasso', 'mrmr', 'pca']:
        if fs_method == 'region_based':
            seq_data_1 = seq_data[(seq_data['chromosome_name'] == 1) & (seq_data['band'] == 'p31.3')]
            X_1 = utils.region_based_fs(X, seq_data_1)
        if fs_method == 'chromosome_based':
            seq_data_1 = seq_data[(seq_data['chromosome_name'] == 4)]
            X_1 = utils.chromosome_based_fs(X, seq_data_1)
        if fs_method == 'pathway_based':
            pathway_list_1 = 'REACTOME_NCAM_SIGNALING_FOR_NEURITE_OUT_GROWTH'
            X_1 = utils.pathway_based_fs(X, {pathway_list_1 : pathway_list[pathway_list_1]})
        if 'based' in fs_method:
            #for feat_name in list(X_1.keys()):
            results_all[fs_method] = utils.traditional_ml_kfold_many(X_groups[fs_method], y['response_NR'], path_kidney,
                                                              smote_flag, 10, fs_method)


        if fs_method == 'lasso' or fs_method == 'mrmr':
            for feat_name in [50, 100, 200, 500, 1000, 2000, 5000]:
                results_all[fs_method+str(feat_name)] = utils.traditional_ml_kfold(X, y['response_NR'], path_kidney,
                                                              smote_flag, feat_name, 10, fs_method)

        if fs_method == 'pca':
            for feat_name in [0.95]:
                results_all[fs_method+str(feat_name)] = utils.traditional_ml_kfold(X, y['response_NR'], path_kidney,
                                                             smote_flag, feat_name, 10, fs_method)


    # In[ ]:
    if smote_flag == 0:
        with open('results_stomach_no_smote_all.pkl', 'wb') as f:
            pickle.dump(results_all, f)
    else:
        with open('results_stomach_smote_all.pkl', 'wb') as f:
            pickle.dump(results_all, f)

    utils.export_results_to_excel(results_all, smote_flag, out_dir=path_kidney)'''
