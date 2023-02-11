import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns
import matplotlib

# load a csv file into a pandas dataframe
def load_csv(filename):
    df = pd.read_csv(filename)
    return df


# TODO: 
# Proper Parser for google benchmark output files, 
# so no hand-editing is needed for the csv files to get rid of header+namingConvetion
# redo the benchmarking of Week1 with gbench
def add_fm_to_bs_ml_l100_nx_df(filename, n, df):
    df_fm_index = load_csv(filename)
    df_fm_index = df_fm_index[df_fm_index['name'] == 'BM_search_fm_index']
    df_fm_index = df_fm_index[["real_time"]] 
    df_fm_index = df_fm_index.rename(columns={"real_time": "FM-Index"})
    # from nanoseconds to seconds
    df_fm_index = df_fm_index / 1000000000
    # add a column 
    df_fm_index["n"] = n
    # add the df_fm_index to the df
    df = pd.concat([df, df_fm_index], axis=0)
    return df

def add_fm_to_all_lx_n10_df(filename, df):
    df_fm_index = load_csv(filename)
    df_fm_index_crop = df_fm_index[["real_time"]] 
    df_fm_index_crop = df_fm_index_crop.rename(columns={"real_time": "FM-Index"})
    df_fm_index_crop = df_fm_index_crop / 1000000000
    df_fm_index_crop["n"] = df_fm_index[["name"]]
    df = pd.concat([df, df_fm_index_crop], axis=0)
    return df

def get_fm_lx_n10000_ky_df(filename):
    # edited the full file to have a column with the #errors and cropped the names
    df_fm_index = load_csv(filename)
    df_fm_index_crop = df_fm_index[["real_time", "name", "error_k"]] 
    df_fm_index_crop = df_fm_index_crop.rename(columns={"real_time": "FM-Index", "name": "n"})
    df_fm_index_crop["FM-Index"] = df_fm_index_crop[["FM-Index"]] / 1000000000
    df_fm_index_crop = df_fm_index_crop.rename(columns={"n": "error_k", "error_k": "n"})
    return df_fm_index_crop


# plot the data
def plot(df, title, xlabel, ylabel, filename):
    plt.clf()
    df = df.sort_values(by=['n'])
    df = df.reset_index(drop=True)
    df = df.melt(id_vars=['n'], var_name='algorithm', value_name='time')
    sns.set(style="whitegrid")
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    sns.set_palette("Set2")
    g1 = sns.catplot(x="n", y="time", hue="algorithm", data=df, kind="box", height=6, aspect=1.5)
    g1 = sns.stripplot(x="n", y="time", hue="algorithm", data=df, dodge=True, legend=False, size=5, alpha=0.8)
    g1.set(xlabel=xlabel, ylabel=ylabel)
    g1.set(title=title)
    # save the plot to filename
    plt.savefig(filename)
    plt.show()
    return

def plot_fm_full(df, title, xlabel, ylabel, filename):
    plt.clf()
    sns.set(style="whitegrid")
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    sns.set_palette("Set2")
    g1 = sns.catplot(x="error_k", y="FM-Index", hue="n", data=df, kind="box", height=6, aspect=1.5)
    g1 = sns.swarmplot(x="error_k", y="FM-Index", hue="n", data=df, dodge=True, legend=False, size=5, alpha=0.8, palette="Set2")
    g1.set(xlabel=xlabel, ylabel=ylabel)
    g1.set(title=title)
    # save the plot to filename
    plt.savefig(filename)
    plt.show()
    return


if __name__ == '__main__':
    df = load_csv('combined_100l_bs_mlr.csv')
    df = add_fm_to_bs_ml_l100_nx_df('fmIndex_100l_n1000_10i.csv', 1000, df)
    df = add_fm_to_bs_ml_l100_nx_df('fmIndex_100l_n10000_10i.csv', 10000, df)
    df = add_fm_to_bs_ml_l100_nx_df('fmIndex_100l_n100000_10i.csv', 100000, df)
    plot(df, 'Running time of suffix array based approaches', '# of queries', 'time (seconds)', 'plot100.png')

    df = load_csv('combined_40l_60l_all10n.csv')
    df = add_fm_to_all_lx_n10_df('fmIndex_n10_i25.csv', df)
    plot(df, 'Running time of all methods', 'Query length', 'time (seconds)', 'plot40_60_80_100.png')

    df = get_fm_lx_n10000_ky_df('fmindex_full_n1000_i10.csv')
    plot_fm_full(df, 'Running time of FM-Index and PEX-Search with errors', 'Query length', 'time (seconds)', 'plotfmindex_all.png')





