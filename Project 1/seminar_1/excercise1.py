# This is just my code .. so you can ignore this cell 
# it is not necessary for the exercise. 
# I was just lazy af


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os, glob, sys, gc
import random # for reproducibility 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# set seeds for reproducibility
def set_seeds(seed = 1129142087):
    np.random.seed(0)
    random.seed(seed)
    np.random.seed(seed)
    print('Seeds set to {}.'.format(seed))
    return

# read the data
def read_data(filename):
    fruits = pd.read_table(filename)
    return fruits

# replace '?' with None
def replace_question_mark(df):
    df.replace('?', None, inplace=True)
    return df

# check the data
def check_data(df):
    print(df.head())
    print(df.info())
    print(df.describe())
    return

# check the missing values
def check_missing(df):
    print("Missing values: ", df.isnull().sum())
    return

# check the correlation
def check_corr(df):
    corr = df.corr()
    print(corr)
    return

# check the distribution of the data
def check_dist(df):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    sns.distplot(df['sepal_length'], ax=ax[0, 0])
    plt.show()
    return

# check the boxplot
def check_boxplot(df):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    sns.boxplot(df['sepal_length'], ax=ax[0, 0])
    plt.show()
    return

# check the pairplot
def check_pairplot(df):
    #sns.pairplot(df, hue='species')
    sns.pairplot(df)
    plt.show()
    return

# check the heatmap
def check_heatmap(df):
    corr = df.corr()
    #sns.heatmap(corr, annot=True)
    sns.heatmap(corr)
    plt.show()
    return

# check the scatterplot
def check_scatterplot(df):
    #sns.scatterplot(x='sepal_length', y='sepal_width', data=df, hue='species')
    sns.scatterplot(x='sepal_length', y='sepal_width', data=df)
    plt.show()
    return


# main function
def main():
    # read the data
    df = read_data('data.txt')
    # replace '?' with None
    df = replace_question_mark(df)
    # check the data
    #check_data(df)
    # check the missing values
    #check_missing(df)
    # check the correlation
    #check_corr(df)
    # check the distribution of the data
    #check_dist(df)
    # check the boxplot
    #check_boxplot(df)
    # check the pairplot
    #check_pairplot(df)
    # check the heatmap
    #check_heatmap(df)
    # check the scatterplot
    #check_scatterplot(df)
    return df


from __future__ import print_function, division
if __name__ == '__main__':
    df = main()
