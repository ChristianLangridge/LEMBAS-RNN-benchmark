### Exploring data to know which GLM I should use for 

install.packages("psych")
install.packages("car")
install.packages("corrplot")

### Loading packages and data
library(tidyverse)
library(dplyr)
library(psych)      # For describe statistics
library(car)        # For VIF
library(corrplot)   # For correlation matrix plot

gene_expression  <- read_tsv("/Users/christianlangridge/Desktop/Zhang-Lab copy 2/Zhang Lab Data/Pocket data files/Geneexpression(pocket).tsv")
tf_expression <- read_tsv("/Users/christianlangridge/Desktop/Zhang-Lab copy 2/Zhang Lab Data/Pocket data files/TF(pocket).tsv")

### Data sets too large to run in R, so I'm conducting random sampling 

set.seed(123) # Random seed for reproducibility

# Function to sample columns representative for analysis
sample_columns <- function(df, sample_size = 20) {
  numeric_cols <- sapply(df, is.numeric)
  numeric_colnames <- names(df)[numeric_cols]
  
  n <- min(sample_size, length(numeric_colnames)) # If sample_size larger than available numeric columns, use all
  
  sampled_cols <- sample(numeric_colnames, n) # Randomly sample column names
  return(df[, sampled_cols, drop = FALSE])
}

# Sample columns from each large dataset
gene_expression_sample <- sample_columns(gene_expression, sample_size = 20)
tf_expression_sample <- sample_columns(tf_expression, sample_size = 20)

### Looking at variance and distribution in both sample files 
### Column-wise distribution and variance, 
### among other stats parameters look completely different between genes

print("Summary Statistics for gene_expression_sample:")
print(describe(gene_expression_sample))    # Includes mean, sd, skewness, kurtosis
print("Variance of columns in gene_expression_sample:")
print(sapply(gene_expression_sample, var))

print("Summary Statistics for tf_expression_sample:")
print(describe(tf_expression_sample))
print("Variance of columns in tf_expression_sample:")
print(sapply(tf_expression_sample, var))

### Making a correlation matrix for both files
### Weak correlation across sampled data 

print("Correlation matrix for gene_expression_sample:")
cor_gene_expression_sample <- cor(gene_expression_sample, use="complete.obs")
print(cor_gene_expression_sample)
corrplot(cor_gene_expression_sample, method="color")

print("Correlation matrix for tf_expression_sample:")
cor_tf_expression_sample <- cor(tf_expression_sample, use="complete.obs")
print(cor_tf_expression_sample)
corrplot(cor_tf_expression_sample, method="color")

### Multicollinearity assessment (VIF) - using all variables in a simple model
### Suppose the first column is the response, rest as predictors
### Very mild multi-collinearity between variables, biologically explainable as co-activation/expression
### or randomly expressed (bulk RNA-seq catching cells at different timepoints of cellular processes)

gene_expression_sample <- as.data.frame(gene_expression_sample)
tf_expression_sample <- as.data.frame(tf_expression_sample)

if (ncol(gene_expression_sample) > 1) {
  model1 <- lm(gene_expression_sample[[1]] ~ ., data = gene_expression_sample)
  print("VIF values gene_expression_sample:")
  print(vif(model1))
}

if (ncol(tf_expression_sample) > 1) {
  model2 <- lm(tf_expression_sample[[1]] ~ ., data = tf_expression_sample)
  print("VIF values tf_expression_sample:")
  print(vif(model2))
}

