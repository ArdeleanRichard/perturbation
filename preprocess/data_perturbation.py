import numpy as np

def permute_columns(data, column_set):
    length = len(data)
    permuted_indexes = np.random.permutation(length)

    new_data = np.copy(data)
    # for id, perm_id in zip(range(len(data)), permuted_indexes):
    for column in column_set:
        test = data[:, column]
        new_data[:, column] = test[permuted_indexes]

    return new_data