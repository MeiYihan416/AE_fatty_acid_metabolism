#!/usr/bin/env python

import scanpy as sc
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colors
import seaborn as sb
from gprofiler import gprofiler
import anndata
import scvelo as scv
import scirpy as ir

sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()

colors2 = plt.cm.Reds(np.linspace(0, 1, 128))
colors3 = plt.cm.Greys_r(np.linspace(0.7,0.8,20))
colorsComb = np.vstack([colors3, colors2])
mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colorsComb)
scv.set_figure_params()

###==============load scRNA-seq datasets and perform quality control===============================
#================control group=========================================
control_adata_to_merge = sc.read_10x_mtx(
    './control/filtered_feature_bc_matrix',  # the directory with the `.mtx` file
    var_names='gene_symbols',                  # use gene symbols for the variable names (variables-axis index)
    cache=True)
sc.pl.highest_expr_genes(control_adata_to_merge, n_top=20, )
# Quality control - calculate QC covariates
control_adata_to_merge.obs['n_counts'] = control_adata_to_merge.X.sum(1)
control_adata_to_merge.obs['log_counts'] = np.log(control_adata_to_merge.obs['n_counts'])
control_adata_to_merge.obs['n_genes'] = (control_adata_to_merge.X > 0).sum(1)
mito_genes = control_adata_to_merge.var_names.str.startswith('mt-')

control_adata_to_merge.obs['mt_frac'] = np.sum(
    control_adata_to_merge[:, mito_genes].X, axis=1).A1 / np.sum(control_adata_to_merge.X, axis=1).A1
control_adata_to_merge.obs['n_counts'] = control_adata_to_merge.X.sum(axis=1).A1
t1 = sc.pl.violin(control_adata_to_merge, 'n_counts', size=2, log=True,cut=0)
t2 = sc.pl.violin(control_adata_to_merge, 'mt_frac')
#Thresholding decision: counts
p3 = sb.distplot(control_adata_to_merge.obs['n_counts'], kde=False)
plt.show()
p4 = sb.distplot(control_adata_to_merge.obs['n_counts'][control_adata_to_merge.obs['n_counts']<10000], kde=False, bins=60)
plt.show()
p5 = sb.distplot(control_adata_to_merge.obs['n_counts'][control_adata_to_merge.obs['n_counts']>20000], kde=False, bins=60)
plt.show()
#Thresholding decision: genes
p6 = sb.distplot(control_adata_to_merge.obs['n_genes'], kde=False, bins=60)
plt.show()
p7 = sb.distplot(control_adata_to_merge.obs['n_genes'][control_adata_to_merge.obs['n_genes']<2000], kde=False, bins=60)
plt.show()
# Filter cells according to identified QC thresholds:
print('Total number of cells: {:d}'.format(control_adata_to_merge.n_obs))
sc.pp.filter_cells(control_adata_to_merge, min_counts = 6000)
print('Number of cells after min count filter: {:d}'.format(control_adata_to_merge.n_obs))
sc.pp.filter_cells(control_adata_to_merge, max_counts = 80000)
print('Number of cells after max count filter: {:d}'.format(control_adata_to_merge.n_obs))
control_adata_to_merge = control_adata_to_merge[control_adata_to_merge.obs['mt_frac'] < 0.15]
print('Number of cells after MT filter: {:d}'.format(control_adata_to_merge.n_obs))
sc.pp.filter_cells(control_adata_to_merge, min_genes = 1200)
print('Number of cells after gene filter: {:d}'.format(control_adata_to_merge.n_obs))



#================mCherry group=========================================
mCherry_adata_to_merge = sc.read_10x_mtx(
    './mCherry/filtered_feature_bc_matrix',  # the directory with the `.mtx` file
    var_names='gene_symbols',                  # use gene symbols for the variable names (variables-axis index)
    cache=True)
sc.pl.highest_expr_genes(mCherry_adata_to_merge, n_top=20, )
# Quality control - calculate QC covariates
mCherry_adata_to_merge.obs['n_counts'] = mCherry_adata_to_merge.X.sum(1)
mCherry_adata_to_merge.obs['log_counts'] = np.log(mCherry_adata_to_merge.obs['n_counts'])
mCherry_adata_to_merge.obs['n_genes'] = (mCherry_adata_to_merge.X > 0).sum(1)
mito_genes = mCherry_adata_to_merge.var_names.str.startswith('mt-')
# for each cell compute fraction of counts in mito genes vs. all genes
# the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
mCherry_adata_to_merge.obs['mt_frac'] = np.sum(
    mCherry_adata_to_merge[:, mito_genes].X, axis=1).A1 / np.sum(mCherry_adata_to_merge.X, axis=1).A1
# add the total counts per cell as observations-annotation to adata
mCherry_adata_to_merge.obs['n_counts'] =mCherry_adata_to_merge.X.sum(axis=1).A1
#Thresholding decision: counts
p3 = sb.distplot(mCherry_adata_to_merge.obs['n_counts'], kde=False)
plt.show()
p4 = sb.distplot(mCherry_adata_to_merge.obs['n_counts'][mCherry_adata_to_merge.obs['n_counts']<6000], kde=False, bins=60)
plt.show()
p5 = sb.distplot(mCherry_adata_to_merge.obs['n_counts'][mCherry_adata_to_merge.obs['n_counts']>15000], kde=False, bins=60)
plt.show()
#Thresholding decision: genes
p6 = sb.distplot(mCherry_adata_to_merge.obs['n_genes'], kde=False, bins=60)
plt.show()
p7 = sb.distplot(mCherry_adata_to_merge.obs['n_genes'][mCherry_adata_to_merge.obs['n_genes']<2000], kde=False, bins=60)
plt.show()
# Filter cells according to identified QC thresholds:
print('Total number of cells: {:d}'.format(mCherry_adata_to_merge.n_obs))
sc.pp.filter_cells(mCherry_adata_to_merge, min_counts = 3000)
print('Number of cells after min count filter: {:d}'.format(mCherry_adata_to_merge.n_obs))
sc.pp.filter_cells(mCherry_adata_to_merge, max_counts = 60000)
print('Number of cells after max count filter: {:d}'.format(mCherry_adata_to_merge.n_obs))
mCherry_adata_to_merge = mCherry_adata_to_merge[mCherry_adata_to_merge.obs['mt_frac'] < 0.15]
print('Number of cells after MT filter: {:d}'.format(mCherry_adata_to_merge.n_obs))
sc.pp.filter_cells(mCherry_adata_to_merge, min_genes = 1250)
print('Number of cells after gene filter: {:d}'.format(mCherry_adata_to_merge.n_obs))

#================non-mCherry group=========================================
nonmCherry_adata_to_merge = sc.read_10x_mtx(
    './non_mCherry/filtered_feature_bc_matrix',  # the directory with the `.mtx` file
    var_names='gene_symbols',                  # use gene symbols for the variable names (variables-axis index)
    cache=True)
sc.pl.highest_expr_genes(nonmCherry_adata_to_merge, n_top=20, )
# Quality control - calculate QC covariates
nonmCherry_adata_to_merge.obs['n_counts'] = nonmCherry_adata_to_merge.X.sum(1)
nonmCherry_adata_to_merge.obs['log_counts'] = np.log(nonmCherry_adata_to_merge.obs['n_counts'])
nonmCherry_adata_to_merge.obs['n_genes'] = (nonmCherry_adata_to_merge.X > 0).sum(1)
mito_genes = nonmCherry_adata_to_merge.var_names.str.startswith('mt-')
# for each cell compute fraction of counts in mito genes vs. all genes
# the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
nonmCherry_adata_to_merge.obs['mt_frac'] = np.sum(
    nonmCherry_adata_to_merge[:, mito_genes].X, axis=1).A1 / np.sum(nonmCherry_adata_to_merge.X, axis=1).A1
# add the total counts per cell as observations-annotation to adata
nonmCherry_adata_to_merge.obs['n_counts'] = nonmCherry_adata_to_merge.X.sum(axis=1).A1
#Thresholding decision: counts
p3 = sb.distplot(nonmCherry_adata_to_merge.obs['n_counts'], kde=False)
plt.show()
p4 = sb.distplot(nonmCherry_adata_to_merge.obs['n_counts'][nonmCherry_adata_to_merge.obs['n_counts']<6000], kde=False, bins=60)
plt.show()
p5 = sb.distplot(nonmCherry_adata_to_merge.obs['n_counts'][nonmCherry_adata_to_merge.obs['n_counts']>15000], kde=False, bins=60)
plt.show()
#Thresholding decision: genes
p6 = sb.distplot(nonmCherry_adata_to_merge.obs['n_genes'], kde=False, bins=60)
plt.show()
p7 = sb.distplot(nonmCherry_adata_to_merge.obs['n_genes'][nonmCherry_adata_to_merge.obs['n_genes']<2000], kde=False, bins=60)
plt.show()
# Filter cells according to identified QC thresholds:
print('Total number of cells: {:d}'.format(nonmCherry_adata_to_merge.n_obs))
sc.pp.filter_cells(nonmCherry_adata_to_merge, min_counts = 3000)
print('Number of cells after min count filter: {:d}'.format(nonmCherry_adata_to_merge.n_obs))
sc.pp.filter_cells(nonmCherry_adata_to_merge, max_counts = 60000)
print('Number of cells after max count filter: {:d}'.format(nonmCherry_adata_to_merge.n_obs))
nonmCherry_adata_to_merge = nonmCherry_adata_to_merge[nonmCherry_adata_to_merge.obs['mt_frac'] < 0.15]
print('Number of cells after MT filter: {:d}'.format(nonmCherry_adata_to_merge.n_obs))
sc.pp.filter_cells(nonmCherry_adata_to_merge, min_genes = 1500)
print('Number of cells after gene filter: {:d}'.format(nonmCherry_adata_to_merge.n_obs))

###=============Intergrate three datasets===============================
control_adata_to_merge.X=control_adata_to_merge.X.toarray()
nonmCherry_adata_to_merge.X=nonmCherry_adata_to_merge.X.toarray()
mCherry_adata_to_merge.X=mCherry_adata_to_merge.X.toarray()
control_adata_to_merge.obs["Group"] = "Control"
nonmCherry_adata_to_merge.obs["Group"] = "non_mCherry"
mCherry_adata_to_merge.obs["Group"] = "mCherry"
adata = control_adata_to_merge.concatenate(nonmCherry_adata_to_merge,mCherry_adata_to_merge,batch_key='Group', batch_categories=["Control",'non_mCherry','mCherry'])
adata.obs_names = [c.split("-")[0] for c in adata.obs_names]
adata.obs_names_make_unique(join='_')
adata.X = adata.X.astype('float64')
adata.raw = adata

###==============Remove doubletsa and preprocess===============================

import scrublet as scr
sc.external.pp.scrublet(adata, batch_key='Group', expected_doublet_rate=0.065)
sc.external.pl.scrublet_score_distribution(adata)
adata = adata[adata.obs["predicted_doublet"] == False]
sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.log1p(adata)
## Highly variable genes ##
sc.pp.highly_variable_genes(adata, n_top_genes=4000)
print('\n','Number of highly variable genes: {:d}'.format(np.sum(adata.var['highly_variable'])))


###==============reduce dimension and clustering===============================

sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.diffmap(adata)
sc.tl.leiden(adata, resolution=1, key_added='leiden_r1', random_state=0)
sc.tl.rank_genes_groups(adata, groupby='leiden_r1', key_added='rank_genes_r1')


###==============cell annotation===============================

#Known marker genes:
marker_genes = dict()
marker_genes['qHSC'] = ['Ifitm1', 'H2-K1', 'Mpl', 'Ifitm3', 'Txnip', 'Mycn','Tgm2','Procr','Socs2', 'Ctla2a', 'Ctla2b', 'H2-K1', 'Lmo2', 'Adgrg1','Esam','Ifitm1','Pdss1','Txnip','Hlf']
marker_genes['aHSC'] = ['Cd34','H2afy','Gas5','Sox4','Pim1','Adgrl4','Myc','Ccl3','Pdss1','Adgrg3','Btg2']
marker_genes['cHSC'] = ['H2-K1', 'Prdx1', 'Lig1', 'Top2a', 'Cdt1', 'Pdss1', 'Pcna', 'Stmn1', 'Tubb5', 'Tuba1b', 'Nrm']
marker_genes['Mk'] = ['Pf4', 'Rap1b', 'Tmsb4x', 'Itga2b', 'Pbx1', 'Cd9', 'Plek', 'Gata2']
marker_genes['GMP'] = ['Mpo', 'Elane', 'Prtn3', 'Plac8', 'Ctsg', 'Gstm1', 'Pgam1', 'Gfi1', 'Cebpe', 'Ly86', 'Csf1r', 'Irf8']
marker_genes['Ery'] = ['Car1', 'Car2', 'Fam132a', 'Aqp1', 'Hspe1', 'Ermap','Rhd','Klf1']
marker_genes['LMP'] = ['Rag2', 'Flt3', 'Dntt','Il12a', 'Satb1']
marker_genes['Mast'] = ['Cma1','Gzmb']
marker_genes['pre-DC'] = ['H2-Aa', 'H2-Eb1', 'Cst3', 'Cd74', 'H2-Ab1', 'Irf8','H2-DMb1']
marker_genes['pre-B'] = ['Vpreb3', 'Vpreb1', 'Ighm', 'Cd79a', 'Dntt', 'Igll1','Ebf1']
marker_genes['Gran']=['S100a8','S100a9','Ngp']

cell_annotation = sc.tl.marker_gene_overlap(adata, marker_genes, key='rank_genes_r1')
cell_annotation_norm = sc.tl.marker_gene_overlap(adata, marker_genes, key='rank_genes_r1', normalize='reference')
sb.heatmap(cell_annotation_norm, cbar=True,xticklabels=1,center=1,fmt='0.01g')

Cluster = {'0':'aHSC_1', 
               "1":'LMP',
               "2":'aHSC_2',
               "3":'GMP',
               "5":'GMP',
               "6":'Ery',
               "7":'Cd4_Tn',
               "8":'qHSC_1',
               "9":'Mk',
               "10":'GMP',
               "11":'Ery',
               "12":'Mk',
               "13":'Mk',
               "14":'cHSC',
               "15":'cHSC',
               "16":'qHSC_2',
               "17":'qHSC_2',
               "18":'qHSC_2'}

adata.obs['Cluster'] = (adata.obs['leiden_r1'].map(Cluster).astype('category'))

from matplotlib import pyplot as plt, cm as mpl_cm
from cycler import cycler
sc.pl.umap(adata, color=['Cluster'], legend_loc='right margin',legend_fontsize=7,palette=cycler(color=mpl_cm.Paired.colors))
sc.pl.umap(adata, color=['Cluster'], legend_loc='on data',legend_fontsize=7,palette=cycler(color=mpl_cm.Paired.colors))
sc.pl.umap(adata, color=['Group'], legend_loc='on data',legend_fontsize=7,palette=cycler(color=mpl_cm.Paired.colors))
sc.pl.umap(adata, color=['Group'], legend_loc='right margin',legend_fontsize=7,palette=cycler(color=mpl_cm.Paired.colors))


sc.tl.rank_genes_groups(adata, groupby='Cluster', method='wilcoxon')
sc.pl.rank_genes_groups_dotplot(adata, n_genes=10,standard_scale='var')

