import numpy as np
from matplotlib import pyplot as plt


def plot_training(training):
    # plot
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.tight_layout(pad=3.0)

    # training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black', label="loss")
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
        ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    ax[0].legend()

    # validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black', label="loss")
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_' + metric], label=metric)
        ax22.set_ylabel("Score", color="steelblue")
    ax22.legend()
    ax[1].legend()

    # PLOT_PATH = f'./figures/neuralNetworks/'
    # filename = "sim" + str(simNr) + "_superlet_ord" + str(ord) + "_ncyc" + str(ncyc)
    #
    # plt.savefig(PLOT_PATH + filename + ".png")

# functie pentru vizualizarea etichetelor inainte si dupa codificare
# pentru a asigura integritatea datelor


def visualize_labels_after_preprocessing(labels, encoded_labels):
    print('Labels shape before preprocessing: ' + str(labels.shape))
    print('Unique labels: ' + str(np.unique(labels)))
    print('First three labels shape: ' + str(labels[0:3]))
    print('---------------------------------------------------------------')
    print('Labels shape after preprocessing: ' + str(encoded_labels.shape))
    print('Unique labels: ' + str(np.unique(encoded_labels)))
    print('First three labels shape: ' + str(encoded_labels[0:3]))


# functie pentru a vizualiza statistici despre cum a fost impartit setul de date
def visualize_data_after_split(slt_features, labels, features_test, labels_test,
                               features_train, labels_train, features_validation, labels_validation):
    print('No of slt features:' + str(len(slt_features)))
    print('Length of each feature: ' + str(len(slt_features[0])))
    print('No of labels:' + str(len(labels)))
    print('---------------------------------------------------------------')
    print('No of slt features for training:' + str(len(features_train)))
    print('No of labels for training:' + str(len(labels_train)))
    print('---------------------------------------------------------------')
    print('No of slt features for validation:' + str(len(features_validation)))
    print('No of labels for validation:' + str(len(labels_validation)))
    print('---------------------------------------------------------------')
    print('No of slt features for testing:' + str(len(features_test)))
    print('No of labels for testing:' + str(len(labels_test)))
