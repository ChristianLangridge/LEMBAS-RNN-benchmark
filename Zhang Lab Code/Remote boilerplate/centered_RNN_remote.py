##### loading and centering data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# reading in full data files
gene_expression = pd.read_csv(('/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Data/Full data files/Geneexpression (full).tsv'), sep='\t', header=0)
tf_expression = pd.read_csv(('/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Data/Full data files/TF(full).tsv'), sep='\t', header=0)

# test-train splitting
x = tf_expression
y = gene_expression

# saving gene IDs
feature_names = tf_expression.columns.tolist()
target_names = gene_expression.columns.tolist()

# first split: 70% train and 30% temp (test + val)
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3, random_state=42)

# second split: 20% test and 10% val (which is 2/3 and 1/3 of temp)
x_test, x_val, y_test, y_val = train_test_split(
    x_temp, y_temp, test_size=1/3, random_state=42)

# converting to numpy for for mean centering
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

x_val = x_val.to_numpy()
y_val = y_val.to_numpy()

x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

# column-wise mean centering for each set (each gene is a column, each row is an instance)
# Applying training means to avoid data leakage
# x centering -> stabilises model numeric conditioning (sensitivity to input qualities)
# y centering -> reduces structural multicollinearity, 
# predictions become representative of mean gene expression, improves generalization by removing absolute scale artifacts

x_train_col_means = x_train.mean(axis=0)
x_train_centered = x_train - x_train_col_means

y_train_col_means = y_train.mean(axis=0)
y_train_centered = y_train - y_train_col_means

# for test set 
x_test_centered = x_test - x_train_col_means
y_test_centered = y_test - y_train_col_means

# for val set
x_val_centered = x_val - x_train_col_means
y_val_centered = y_val - y_train_col_means

# converting back into a dataframe to restore gene IDs 
x_train_centered_df = pd.DataFrame(x_train_centered, columns=feature_names)
x_val_centered_df = pd.DataFrame(x_val_centered, columns=feature_names)
x_test_centered_df = pd.DataFrame(x_test_centered, columns=feature_names)

y_train_centered_df = pd.DataFrame(y_train_centered, columns=target_names)
y_val_centered_df = pd.DataFrame(y_val_centered, columns=target_names)
y_test_centered_df = pd.DataFrame(y_test_centered, columns=target_names)

# saving column means for potential future usage
x_train_column_means_df = pd.DataFrame([x_train_col_means], columns=feature_names)
y_train_column_means_df = pd.DataFrame([y_train_col_means], columns=target_names)

