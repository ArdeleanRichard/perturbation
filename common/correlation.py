import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


def find_correlated_sets_from_matrix(corr_matrix, threshold):
    # correlate sets are found by iterating one by one each characteristic
    # and creating a list that has all other characteristics above the threshold
    sets = []
    for row in corr_matrix:
        above_threshold = []
        for i, val in enumerate(row):
            if val >= threshold:
                above_threshold.append(i)
        sets.append(above_threshold)
    return sets


def get_correlation_matrix(data, show=True, plot_path=None, filename=None):
    df = pd.DataFrame(data)
    matrix = np.array(df.corr())

    # print(matrix)

    if show == True:
        fig, ax = plt.subplots(figsize=(8, 5))
        _ = sns.heatmap(matrix, cmap="coolwarm")
        plt.savefig(plot_path + filename + ".png")
        plt.show()

    return matrix


def get_correlated_sets(data, threshold, show=True, plot_path=None, filename=None):
    matrix = get_correlation_matrix(data, show=True, plot_path=plot_path, filename=filename)
    correlated_sets = find_correlated_sets_from_matrix(matrix, threshold=threshold)

    if show == True:
        for i in range(0, len(correlated_sets)):
            print(correlated_sets[i])

    return correlated_sets

