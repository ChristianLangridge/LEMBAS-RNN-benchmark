import pandas as pd
import numpy as np
import shap 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import pickle 
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
import multiprocessing as mp
from sklearn.model_selection import cross_val_score
from multiprocessing import Manager
import optuna
import glob
import os
import sys
import json
from pathlib import Path

# Resolve REPO_ROOT and DATA_ROOT
# Works whether the script is run directly (python script.py) 
# or via %run from a notebook
if 'REPO_ROOT' not in dir():
    _root = next(p for p in Path(__file__).resolve().parents if (p / "README.md").exists())
    REPO_ROOT = str(_root)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if 'DATA_ROOT' not in dir():
    with open(Path(REPO_ROOT) / "data_config.json", "r") as f:
        DATA_ROOT = json.load(f)["DATA_ROOT"]

# Reading in full data files
gene_expression = pd.read_csv((f"{DATA_ROOT}/Full data files/Geneexpression (full).tsv"), sep='\t', header=0, index_col=0)
tf_expression = pd.read_csv((f"{DATA_ROOT}/Full data files/TF(full).tsv"), sep='\t', header=0, index_col=0)

# Making sure only TFs that are in the network are also in the expression data 
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
network_tfs = set(net['TF'].unique())      # TFs
network_genes = set(net['Gene'].unique())  # target genes
network_nodes = network_tfs | network_genes  # all nodes in the network.tsv
usable_features = [tf for tf in tf_expression.columns if tf in network_nodes]

x = tf_expression[usable_features]  # aligned with tf nodes in network.tsv
y = gene_expression

# 80% train and 20% test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=888) # changed from 42 to 888 to match training seed for RNN 13/01/26

# For training set
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

# For testing set
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

