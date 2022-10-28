import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os, glob, sys, gc
import random # for reproducibility 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# set seeds for reproducibility
def set_seeds(seed = 1129142087):
    np.random.seed(0)
    random.seed(seed)
    np.random.seed(seed)
    print('Seeds set to {}.'.format(seed))
    return

# read the data
def read_data(filename):
    df = pd.read_csv(filename)
    return df

# replace '?' with None
def impute_question_marks(df):
    df.replace('?', None, inplace=True)
    # convert data type of all columns to float
    df = df.astype(float)
    # goal is a categorical variable, so convert it to str for better visualization in pairplot etc.
    df.goal = df.goal.astype(str)
    print("Missing values: ", df.isnull().sum())
    print("Duplicate values: ", df.duplicated().sum())
    # impute missing values with mean
    df.fillna(df.mean(), inplace=True)
    return df

# check the data
def check_data(df):
    print(df.head())
    print(df.info())
    print(df.describe())
    return

# check the correlation
def check_corr(df):
    corr = df.corr()
    print(corr)
    return

# check the distribution of the data
def check_dist(df):
    sns.countplot(x='goal', data=df)
    plt.show()
    return

# check the pairplot
def check_pairplot(df):
    sns.pairplot(df, hue='goal')
    plt.show()
    return

# check the heatmap
def check_heatmap(df):
    corr = df.corr()
    sns.heatmap(corr)
    plt.show()
    return

# split the data into train and test
# return train and test dataframes
def split_data(df):
    return train_test_split(df, test_size=0.2)



# TODO: 
# why no PCA?
# Task 2: Set up some classifiers and evaluate them
# Evaluation: Accuracy, Sensitivity&Sepcificity (why not Precision&Recall?), F1-Score for unbalanced data

# multiple linear regression 
def multiple_linear_regression(train, test):
    return 

# k-NN
def k_nearest_neighbors(train, test):
    return

# decision tree
def decision_tree(train, test):
    return

# TODO: 
# task 3: same again, but all !=0 goal-values are set to 1. No need for F1-Score anymore.
# task 4: ROC(&AUC?) curves
# task 5: Report and presentation 

# ------------------ main ------------------
# main function
def main():
    # read the data
    df = read_data('./../data\processedWithHeader.cleveland.data')
    # replace '?' with None, check for missings, and impute missing values with mean
    df = impute_question_marks(df)
    # check the data
    check_data(df)
    # check the correlation
    check_corr(df)
    # check the distribution of the data regarding the goal
    check_dist(df)
    # check the heatmap
    check_heatmap(df)
    # check the pairplot
    check_pairplot(df)
    return df


from __future__ import print_function, division
if __name__ == '__main__':
    df = main()