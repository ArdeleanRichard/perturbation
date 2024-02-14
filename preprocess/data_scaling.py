import numpy as np
from sklearn import preprocessing


# DOES NOT NORMALIZE EACH DIMENSION NIGGA BETWEEN 0 and 1, NORMALIZES THE WHOLE DATA
def normalize_data_min_max(data):
    return (data - np.amin(data)) / (np.amax(data) - np.amin(data))

def normalize_data_z_score(data):
    return (data - np.mean(data)) / np.std(data)