import os

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.utils.np_utils import to_categorical

import neural_networks.ann.apply_nn as nn

from common.bicubic import bicubic
from common.correlation import get_correlated_sets

from data_parsing.SsdParser import SsdParser
from frequency_domain.superlet.apply_slt import generate_spectrogram, plot_spectrogram
from constants import CHANNEL, RUNS, THRESHOLD, PLOT_PATH, DATASET_PATH

from preprocess.data_perturbation import permute_columns
from preprocess.data_scaling import normalize_data_z_score

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def plot_average_spikes(data, labels, ord, ncyc):
    for unique_label in np.unique(labels):
        selected_spikes = data[labels == unique_label]
        average_spike = np.mean(selected_spikes, axis=0)

        filename = PLOT_PATH + f"_average_spike{unique_label}.png"
        generate_spectrogram(average_spike, ncyc, ord, None, label=None, time_measure='samples', show=True,
                             title_sig='Average Spike', title_spec='Average Spike Spectrogram', save=True,
                             filename=filename)


def bar_plot(original_set, disturbed_set, ord, ncyc, title, suptitle):
    # Crearea diagrama de bare
    plt.bar(np.arange(len(original_set)), original_set, width=0.35, label='Perturbed set')
    plt.bar(np.arange(len(disturbed_set)) + 0.35, disturbed_set, width=0.35, label='Original set')

    plt.xticks(np.arange(0, len(original_set), 9) + 0.35 / 2, [str(i + 1) for i in range(0, len(original_set), 9)])

    plt.title(title)

    plt.legend()
    plt.savefig(PLOT_PATH + f"_barplot_{suptitle}_ord{ord}_ncyc_{str(ncyc)}_RUNS{RUNS}_thr{str(THRESHOLD)}.png")

    plt.show()

def calculate_delta(orig, pert):
    return orig - pert

def perturbation(data, labels, ord, ncyc):
    plot_average_spikes(data, labels, ord, ncyc)
    print("*Saved average spikes")

    slt_features = []
    slt_features_vector = []

    # preprocess data, generate spectrogram, apply bicubic interpolation, flatten
    for spike in data:
        spectrogram = generate_spectrogram(spike, ncyc, ord, None)
        interp_spectrogram = bicubic(img=np.expand_dims(spectrogram, axis=2), ratio=0.25, a=-1 / 2).squeeze()
        slt_features.append(interp_spectrogram)
        slt_features_vector.append(interp_spectrogram.flatten())

    slt_features_vector = normalize_data_z_score(slt_features_vector)
    slt_features = np.array(slt_features)
    slt_features_vector = np.array(slt_features_vector)
    print("*Created SLT features")

    encoded_labels = to_categorical(labels.reshape(-1, 1).astype(int))
    encoded_labels = np.array(encoded_labels)

    acc_orig_data = 0
    f1_orig_data = 0

    # Run NN on original data as control
    model, weights = nn.build_neural_network(slt_features_vector.shape[-1], encoded_labels.shape[-1])
    for num in range(RUNS):
        _, metrics = nn.apply_neural_network2(model, weights, slt_features_vector, encoded_labels)
        acc_orig_data += metrics[1]
        f1_orig_data += metrics[4]
        print(f"NN running on original data, RUNS: {num+1}/{RUNS}")
    acc_orig_data = acc_orig_data / RUNS
    f1_orig_data = f1_orig_data / RUNS
    print(f"Average accuracy (across {RUNS} runs) is - ORIG: {acc_orig_data*100:.2f}")
    print(f"Average f1 score (across {RUNS} runs) is - ORIG: {f1_orig_data*100:.2f}")
    print("*NN finished on original data")

    # retrieve correlated sets above a given threshold
    correlated_sets = get_correlated_sets(slt_features_vector, threshold=THRESHOLD, show=False, plot_path=PLOT_PATH, filename=f"_correlation_matrix")
    print("*Found correlated sets of features")

    accs_pert_data = []
    f1s_pert_data = []
    delta_f1s = []
    delta_accs = []

    # take each set of correlated features, perturb them, run NN on perturbed data
    for id, correlated_set in enumerate(correlated_sets):
        acc_pert_data = 0
        f1_pert_data = 0

        # perturb feature RUNS times
        for num in range(RUNS):
            perturbed_features = permute_columns(np.copy(slt_features_vector), column_set=correlated_set)
            _, metrics = nn.apply_neural_network2(model, weights, perturbed_features, encoded_labels)
            acc_pert_data += metrics[1]
            f1_pert_data += metrics[4]
        acc_pert_data = acc_pert_data / RUNS
        f1_pert_data = f1_pert_data / RUNS

        accs_pert_data.append(acc_pert_data)
        f1s_pert_data.append(f1_pert_data)

        print(f"(CS {id+1}/{len(correlated_sets)}) Average accuracy (across {RUNS} runs) is - ORIG: {acc_orig_data*100:.2f}, PERT: {acc_pert_data*100:.2f}")
        print(f"(CS {id+1}/{len(correlated_sets)}) Average f1 score (across {RUNS} runs) is - ORIG: {f1_orig_data*100:.2f}, PERT: {f1_pert_data*100:.2f}")

        delta_f1 = calculate_delta(f1_orig_data, f1_pert_data)
        delta_acc = calculate_delta(acc_orig_data, acc_pert_data)
        delta_f1s.append(delta_f1)
        delta_accs.append(delta_acc)

    # plot bar
    bar_plot(f1s_pert_data, [f1_orig_data] * len(correlated_sets), ord, ncyc, title="F1 score", suptitle="f1s")
    bar_plot(accs_pert_data, [acc_orig_data] * len(correlated_sets), ord, ncyc, title="Accuracy", suptitle="accs")

    delta_f1s_reshaped = np.array(delta_f1s).reshape((slt_features.shape[1], slt_features.shape[2]))
    delta_accs_reshaped = np.array(delta_accs).reshape((slt_features.shape[1], slt_features.shape[2]))

    delta_f1s_expanded = bicubic(img=np.expand_dims(delta_f1s_reshaped, axis=2), ratio=1 / 0.25, a=-1 / 2).squeeze()
    delta_accs_expanded = bicubic(img=np.expand_dims(delta_accs_reshaped, axis=2), ratio=1 / 0.25, a=-1 / 2).squeeze()

    plot_spectrogram(spectrogram=delta_f1s_expanded, signal=data[0], sampling_frequency=32000, fspace=(300, 7000), show=True,
                     title="F1 score",
                     save=True, filename=PLOT_PATH + f"_f1s_ord{ord}_ncyc_{str(ncyc)}_RUNS{RUNS}_thr{str(THRESHOLD)}.png")
    plot_spectrogram(spectrogram=delta_accs_expanded, signal=data[0], sampling_frequency=32000, fspace=(300, 7000), show=True,
                     title="Accuracy",
                     save=True, filename=PLOT_PATH + f"_accs_ord{ord}_ncyc_{str(ncyc)}_RUNS{RUNS}_thr{str(THRESHOLD)}.png")


def run_m_data():
    parser = SsdParser(DATASET_PATH)
    units_by_channels, labels_by_channels, timestamps_by_channels = parser.units_by_channel, parser.labels_by_channel, parser.timestamps_by_channel

    data = np.array(units_by_channels[CHANNEL])
    labels = np.array(labels_by_channels[CHANNEL])

    perturbation(data, labels, ord=2, ncyc=1.5)



def run_k_data():
    parser = SsdParser(DATASET_PATH)

    data, labels, intracellular_labels = parser.read_kampff_channel(CHANNEL)

    perturbation(data, intracellular_labels, ord=2, ncyc=1.5)


if __name__ == "__main__":
    # run_m_data()
    run_k_data()



