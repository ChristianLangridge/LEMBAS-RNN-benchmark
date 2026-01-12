import pandas as pd
import numpy as np
import shap 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import pickle 
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
import os
import multiprocessing as mp
from sklearn.model_selection import cross_val_score
from multiprocessing import Manager
import optuna
import glob

# reading in full data files
# already normalised into TPM, then log2fold transformed
gene_expression = pd.read_csv(('/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Data/Full data files/Geneexpression (full).tsv'), sep='\t', header=0)
tf_expression = pd.read_csv(('/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Data/Full data files/TF(full).tsv'), sep='\t', header=0)

# column-wise centering (each gene is a column, each row is an instance)
gene_expression_col_means = gene_expression.mean(axis=0)
gene_expression_centered = gene_expression.subtract(gene_expression_col_means, axis=1)

tf_expression_col_means = tf_expression.mean(axis=0)
tf_expression_centered = tf_expression.subtract(tf_expression_col_means, axis=1)

# column-wise min-max scaling

#def min_max_scale_col(col):
    #return (col - col.min()) / (col.max() - col.min())  

# for target-gene expression
#gene_col_mins = gene_expression_centered.min(axis=0)  
#gene_col_maxs = gene_expression_centered.max(axis=0)   

# for tf-expression 
#tf_col_mins = tf_expression_centered.min(axis=0)  
#tf_col_maxs = tf_expression_centered.max(axis=0)  

#gene_expression_scaled = gene_expression_centered.apply(min_max_scale_col, axis=0)
#tf_expression_scaled = tf_expression_centered.apply(min_max_scale_col, axis=0)