#from __future__ import print_function, division
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os, glob, sys, gc
import random # for reproducibility 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, plot_roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# set seeds for reproducibility
def set_seeds(seed = 1129142087):
    random.seed(seed)
    np.random.seed(seed+1)
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
    #print(df.head())
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
    print("Pairplot - all features:\n Warning: This may take a while...\n Note: No need for histograms, because they are included in the pairplot (diagonal)") 
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
def split_data(df, test_size=0.2, random_state=12542068):
    train_df = df.sample(frac=1-test_size, random_state=random_state)
    test_df = df.drop(train_df.index)
    return train_df, test_df

def get_features_and_labels(df, features=[], label="goal", use_MM_scaler=True):
    # split the data into train and test
    # return train and test dataframes
    train_df, test_df = split_data(df)
    if not features:
        train_features = train_df.drop(columns=label)
        test_features = test_df.drop(columns=label)
    else: 
        train_features = train_df[features]
        test_features = test_df[features]
    train_labels = train_df[label]
    test_labels = test_df[label]
    # Scale the data to (0..1)
    if use_MM_scaler:
        scaler = MinMaxScaler()
        _ = scaler.fit_transform(train_df)
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
    return train_features, train_labels, test_features, test_labels

# why no PCA? I mean, srsly?! i would be a great introduction to the data, 
# but with low correlation comes lower interpretability of pcas, or something like that. 
# Task 2: Set up some classifiers and evaluate them
# Evaluation as plots: Accuracy, Sensitivity&Sepcificity (aka Precision&Recall?), 
# NOTE: please use the F1-Score for unbalanced classes 
def logistic_regression(train_features, train_labels):
    # apply logistic regression
    logreg = LogisticRegression()
    logreg.fit(train_features, train_labels)
    print("--- Model type:", type(logreg), " ---")
    print("Accuracy (train): ", logreg.score(train_features, train_labels))
    return logreg

# k-NN
def k_nearest_neighbors(train_features, train_labels):
    # apply k-nn
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_features, train_labels)
    print("--- Model type:", type(knn), " ---")
    print("Accuracy (train): ", knn.score(train_features, train_labels))
    return knn

# decision tree
def decision_tree(train_features, train_labels):
    dec_tree = DecisionTreeClassifier()
    dec_tree.fit(train_features, train_labels)
    print("--- Model type:", type(dec_tree), " ---")
    print("Accuracy (train): ", dec_tree.score(train_features, train_labels))
    return dec_tree

# neural network
def neural_network(train_features, train_labels):
    # not in use for now. wrong hyperparameters. But there exists a neural network in sklearn
    resnet = MLPClassifier(hidden_layer_sizes=(32, 32, 16), max_iter=100)
    resnet.fit(train_features, train_labels)
    print("--- Model type:", type(resnet), " ---")
    print("Accuracy (train): ", resnet.score(train_features, train_labels))
    return resnet


def test_model(model, test_features, test_labels):
    # predict the test data
    pred = model.predict(test_features)
    # evaluate the model
    print("Accuracy (test): ", accuracy_score(test_labels, pred))
    print("Confusion Matrix: \n", confusion_matrix(test_labels, pred))
    print("Classification Report: \n", classification_report(test_labels, pred))
    # heatmap of confusion matrix
    sns.heatmap(confusion_matrix(test_labels, pred), annot=True, fmt='d')
    plt.show()
    return pred

def plot_accuracies(models, model_names, train_features, train_labels, test_features, test_labels):
    # Plot the accuracy of the models in a bar chart
    # use train and test accuracy for the bar chart (for each model)
    train_acc = [model.score(train_features, train_labels) for model in models]
    test_acc = [accuracy_score(test_labels, model.predict(test_features)) for model in models]
    overall_acc = pd.DataFrame({"Model type": model_names, "Accuracy (Training)": train_acc, "Accuracy (Test)": test_acc})
    overall_acc.plot.bar(x="Model type", y=["Accuracy (Training)", "Accuracy (Test)"])  # type: ignore
    plt.show()

def plot_precision(models, model_names, test_features, test_labels):
    # Plot the precision of the models in a bar chart
    # use test precision for the bar chart (for each model)
    precision = [precision_score(test_labels, model.predict(test_features), average="weighted") for model in models]
    overall_precision = pd.DataFrame({"Model type": model_names, "Precision": precision})
    overall_precision.plot.bar(x="Model type", y="Precision")  # type: ignore
    plt.show()

def plot_recall(models, model_names, test_features, test_labels):
    # Plot the recall of the models in a bar chart
    # use test recall for the bar chart (for each model)
    recall = [recall_score(test_labels, model.predict(test_features), average="weighted") for model in models]
    overall_recall = pd.DataFrame({"Model type": model_names, "Recall": recall})
    overall_recall.plot.bar(x="Model type", y="Recall")  # type: ignore
    plt.show()

def plot_acc_prec_rec(models, model_names, train_features, train_labels, test_features, test_labels):
    # Plot the accuracy, precision and recall of the models in a bar chart
    # use train and test accuracy, precision and recall for the bar chart (for each model)
    train_acc = [model.score(train_features, train_labels) for model in models]
    test_acc = [accuracy_score(test_labels, model.predict(test_features)) for model in models]
    precision = [precision_score(test_labels, model.predict(test_features), average="weighted") for model in models]
    recall = [recall_score(test_labels, model.predict(test_features), average="weighted") for model in models]
    overall_acc_prec_rec = pd.DataFrame({"Model type": model_names, "Accuracy (Training)": train_acc, "Accuracy (Test)": test_acc, "Precision (Test)": precision, "Recall (Test)": recall})
    overall_acc_prec_rec.plot.bar(x="Model type", y=["Accuracy (Training)", "Accuracy (Test)", "Precision (Test)", "Recall (Test)"])  # type: ignore
    plt.show()

# TODO: all ROC-Curves in one plot! 
# -> for the report 
def roc_auc(models, model_names, test_features, test_labels):
    # Plot a ROC curve for each model (use test data) 
    # Add the AUC to the plot
    for model in tqdm(models, desc="Plotting ROC-AUC"):
        plot_roc_curve(model, test_features, test_labels)
        # Add the model name to the plot
        plt.title(model_names[models.index(model)])
        plt.show()


# TODO: run it again after binarization of goal 
# task 3: same again, but all !=0 goal-values are set to 1. No need for F1-Score anymore.
# makes sense, 'cause binary classification is easier than multi-class classification, if the features tend to be the same for all goal > 1
def set_goal_to_binary(df):
    df.goal = df.goal.apply(lambda x: "1" if x!="0.0" else "0")
    df.goal = df.goal.astype("int")
    return df

# task 5: Report and presentation 

# ------------------ main ------------------
# main function
if __name__ == '__main__':
    # read the data
    df = read_data('./../data/processedWithHeader.cleveland.data')
    # replace '?' with None, check for missings, and impute missing values with mean
    df = impute_question_marks(df)
    print("Missing values: got imputed with mean values")
    
    # Task 3:
    # Set this to 'True' for task 3 and run the whole code again
    binary_outcome=True
    if binary_outcome:
        df = set_goal_to_binary(df)
    
    # Task 1: Data exploration
    print("-----Task 1: Data exploration-----")
    # check the data
    check_data(df)
    # check the correlation
    check_corr(df)
    # check the distribution of the data regarding the goal
    check_dist(df)
    # check the heatmap
    check_heatmap(df)
    # check the pairplot (this takes a while!)
    check_pairplot(df)

    # Task 2: Set up some classifiers and evaluate them
    print("-----Task 2: Set up some classifiers and evaluate them-----")
    train_features, train_labels, test_features, test_labels = get_features_and_labels(df)

    # knn classifier
    knn = k_nearest_neighbors(train_features, train_labels)
    # evaluate the model
    knn_pred = test_model(knn, test_features, test_labels)

    # logistic regression
    logreg = logistic_regression(train_features, train_labels)
    # evaluate the model
    logreg_pred = test_model(logreg, test_features, test_labels)

    # decision tree
    dec_tree = decision_tree(train_features, train_labels)
    # evaluate the model
    dec_tree_pred = test_model(dec_tree, test_features, test_labels)

    # plot the accuracies of the models
    models = [knn, logreg, dec_tree]
    model_names = ["k-NN", "Log. Reg.", "Decision Tree"]
    plot_accuracies(models, model_names, train_features, train_labels, test_features, test_labels)
    # plot the precision of the models
    plot_precision(models, model_names, test_features, test_labels)
    # plot the recall of the models
    plot_recall(models, model_names, test_features, test_labels)

    plot_acc_prec_rec(models, model_names, train_features, train_labels, test_features, test_labels)

    # Task 4: ROC curves
    print("-----Task 4: ROC curves for a binary outcome-----")
    # plot the ROC-AUC of the models
    if binary_outcome:
        roc_auc(models, model_names, test_features, test_labels)

    # Task 5: Set up a report-template
    print("-----Task 5: Overleaf-Template is set up-----")






