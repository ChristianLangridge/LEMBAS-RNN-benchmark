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
gene_expression = pd.read_csv(('~/Zhang-Lab/Zhang Lab Data/Full data files/Geneexpression (full).tsv'), sep='\t', header=0)
tf_expression = pd.read_csv(('~/Zhang-Lab/Zhang Lab Data/Full data files/TF(full).tsv'), sep='\t', header=0)


# Split into training, testing and validation sets and into numpy arrays + combining dataframes
x = tf_expression
y = gene_expression

combined_data = pd.concat([x, y], axis=1)

# First split: 70% train and 30% temp (test + val)
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3, random_state=888) # changed from 42 to 888 to match training seed for RNN 13/01/26

# Second split: split the temp set into 20% test and 10% val (which is 2/3 and 1/3 of temp)
x_test, x_val, y_test, y_val = train_test_split(
    x_temp, y_temp, test_size=1/3, random_state=888) # changed from 42 to 888 to match training seed for RNN 13/01/26

# For training set
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

# For validation set
x_val = x_val.to_numpy()
y_val = y_val.to_numpy()

# For testing set
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()