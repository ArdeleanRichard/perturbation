import numpy as np

from common.bicubic import bicubic
from common.correlation import get_correlated_sets
from data_parsing.SsdParser import SsdParser
from frequency_domain.superlet.apply_slt import generate_spectrogram, plot_spectrogram
from constants import RUNS, THRESHOLD, PLOT_PATH, ncyc, ord, CHANNEL, DATASET_PATH
from preprocess.data_perturbation import permute_columns

parser = SsdParser(DATASET_PATH)
units_by_channels, labels_by_channels, timestamps_by_channels = parser.units_by_channel, parser.labels_by_channel, parser.timestamps_by_channel

data = np.array(units_by_channels[CHANNEL])
labels = np.array(labels_by_channels[CHANNEL])

slt_features = []
slt_features_vector = []

for spike in data[:3]:
    spectrogram = generate_spectrogram(spike, ncyc, ord, None)

    interp_spectrogram = bicubic(img=np.expand_dims(spectrogram, axis=2), ratio=0.25, a=-1 / 2).squeeze()
    plot_spectrogram(interp_spectrogram, spike, sampling_frequency=32000, fspace=(300, 7000, 50), label=None, time_measure='samples')
    slt_features.append(interp_spectrogram)
    slt_features_vector.append(interp_spectrogram.flatten())

# slt_features = slt.slt(data, ord, ord, ncyc)
slt_features = np.array(slt_features)
print(slt_features.shape)
slt_features_vector = np.array(slt_features_vector)
correlated_sets = get_correlated_sets(slt_features_vector, threshold=THRESHOLD, show=False, plot_path=PLOT_PATH, filename=f"matrice_de_corelatie_C{CHANNEL}")

print(correlated_sets)


def find_max_list_idx(list):
    list_len = [len(i) for i in list]
    return np.argmin(np.array(list_len))


id = find_max_list_idx(correlated_sets)


new_features = permute_columns(slt_features_vector, column_set=correlated_sets[15])

print(new_features.shape)

for vec, spike in zip(new_features, data[:3]):
    hmm = np.reshape(vec, (12, 14))
    plot_spectrogram(hmm, spike, sampling_frequency=32000, fspace=(300, 7000, 50), label=None, time_measure='samples')