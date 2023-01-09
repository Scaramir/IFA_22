import Exc4Task1a as task1a
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# set seed for reproducibility
np.random.seed(0)


def plot_boxplot(c2_list,c3_list,c4_list,c5_list, normalized = False, task = ""):
    title = "Boxplots for parameter estimation"
    if normalized:
        title += " (normalized)"
        c2_list = (c2_list - np.mean(c2_list)) / np.std(c2_list)
        c3_list = (c3_list - np.mean(c3_list)) / np.std(c3_list)
        c4_list = (c4_list - np.mean(c4_list)) / np.std(c4_list)
        c5_list = (c5_list - np.mean(c5_list)) / np.std(c5_list)
    data = [c2_list,c3_list,c4_list,c5_list]
    plt.figure()
    sns.boxplot(data=data, fliersize = 2)
    sns.stripplot(data=data, size=5, jitter=False, linewidth=0.5, edgecolor="gray")
    plt.xticks([0,1,2,3],["c2","c3","c4","c5"])
    plt.xlabel("parameters")
    plt.ylabel("values")
    plt.title(title)
    # save figure
    save_title = "Exc4Task1" + task if normalized == False else "Exc4Task1" + task + "_normalized"
    plt.savefig(save_title + ".png")
    return

if __name__ == "__main__":
    # Task b:
    c2_list = []
    c3_list = []
    c4_list = []
    c5_list = []

    for i in range(0,30):
        popt, pcov  = task1a.easy_curve_fit(np.random.rand(4))
        c2_list.append(popt[0])
        c3_list.append(popt[1])
        c4_list.append(popt[2])
        c5_list.append(popt[3])
        
    plot_boxplot(c2_list,c3_list,c4_list,c5_list, normalized = True, task = "b")
    plot_boxplot(c2_list,c3_list,c4_list,c5_list, task = "b")

    c2_list.clear()
    c3_list.clear()
    c4_list.clear()
    c5_list.clear()

    # Task c: 
    for i in range(0, 30):
        popt, pcov = task1a.easy_curve_fit(
            [np.random.rand(), np.random.rand(), 10, np.random.rand()])
        c2_list.append(popt[0])
        c3_list.append(popt[1])
        c4_list.append(popt[2])
        c5_list.append(popt[3])
    plot_boxplot(c2_list,c3_list,c4_list,c5_list, normalized = True, task = "c")
    plot_boxplot(c2_list,c3_list,c4_list,c5_list, task = "c")
