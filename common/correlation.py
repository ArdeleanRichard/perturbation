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


def get_correlated_sets2(data, threshold, show=False, plot_path=None, filename=None):
    matrix = get_correlation_matrix(data, show=True, plot_path=plot_path, filename=filename)
    correlated_sets = find_correlated_sets_my_way(matrix, threshold=threshold)

    if show == True:
        for i in range(0, len(correlated_sets)):
            print(correlated_sets[i])

    return correlated_sets


def find_correlated_sets_my_way(corr_matrix, threshold):
    groups = []
    for i in range(len(corr_matrix)):
        # groups[f'gr{i}'] = [i]
        groups.append([i])

    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if corr_matrix[i][j] > threshold:
                for group1 in groups:
                    if i in group1:
                        for group2 in groups:
                            if j in group2:
                                if group1 == group2:
                                    break
                                # print(group1, group2)
                                group1.extend(group2)
                                groups.remove(group2)
                                break
                        break
    return groups