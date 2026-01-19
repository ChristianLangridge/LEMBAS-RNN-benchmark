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

# Reading in full data files
gene_expression = pd.read_csv(('~/Zhang-Lab/Zhang Lab Data/Full data files/Geneexpression (full).tsv'), sep='\t', header=0, index_col=0)
tf_expression = pd.read_csv(('~/Zhang-Lab/Zhang Lab Data/Full data files/TF(full).tsv'), sep='\t', header=0, index_col=0)

# Making sure only TFs that are in the network are also in the expression data 
net = pd.read_csv('/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/network(full).tsv', sep='\t')
network_tfs = set(net['TF'].unique())      # TFs
network_genes = set(net['Gene'].unique())  # target genes
network_nodes = network_tfs | network_genes  # all nodes in the network.tsv
usable_features = [tf for tf in tf_expression.columns if tf in network_nodes]

x = tf_expression[usable_features]  # aligned with tf nodes in network.tsv
y = gene_expression

# 80% train and 20% test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=888) # changed from 42 to 888 to match training seed for RNN 13/01/26


#x_train.to_csv("/home/christianl/Zhang-Lab/Zhang Lab Data/Data for Cheng RNN retraining/training/x_train(Christian).csv", sep='\t', encoding='utf-8', index=True)
#y_train.to_csv("/home/christianl/Zhang-Lab/Zhang Lab Data/Data for Cheng RNN retraining/training/y_train(Christian).csv", sep='\t', encoding='utf-8', index=True)

#x_test.to_csv("/home/christianl/Zhang-Lab/Zhang Lab Data/Data for Cheng RNN retraining/testing/x_test(Christian).csv", sep='\t', encoding='utf-8', index=True)
#y_test.to_csv("/home/christianl/Zhang-Lab/Zhang Lab Data/Data for Cheng RNN retraining/testing/y_test(Christian).csv", sep='\t', encoding='utf-8', index=True)

#x_val.to_csv("/home/christianl/Zhang-Lab/Zhang Lab Data/Data for Cheng RNN retraining/validation/x_valChristian).csv", sep='\t', encoding='utf-8', index=True)
#y_val.to_csv("/home/christianl/Zhang-Lab/Zhang Lab Data/Data for Cheng RNN retraining/validation/y_val(Christian).csv", sep='\t', encoding='utf-8', index=True)


# For training set
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

# For validation set
x_val = x_val.to_numpy()
y_val = y_val.to_numpy()

# For testing set
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

