from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.backend import sum

import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


from neural_networks.ann.build_nn import choose_nn
from neural_networks.ann.vis_nn import plot_training
from validation.classification_metrics import F1, precision, recall




def build_neural_network(input_shape, output_shape,
                         nn_type="softmax",
                         optimizer='adam',
                         loss='categorical_crossentropy'
                         ):
    model = choose_nn(nn_type, n_features=input_shape, output_size=output_shape)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', precision, recall, F1])

    return model, model.get_weights()



def apply_neural_network2(model, weights, slt_features, encoded_labels,
                         epochs=100,
                         batch_size=32,
                         verbose=0,
                         show=False,
                         weighted=False
                         ):

    model.set_weights(weights)

    # visualize_labels_after_preprocessing(labels, encoded_labels)

    # Separate the test data
    features, features_test, encoded_labels, encoded_labels_test = train_test_split(slt_features,
                                                                                    encoded_labels, test_size=0.15,
                                                                                    shuffle=True)

    # Split the remaining data to train and validation
    features_train, features_validation, labels_train, labels_validation = train_test_split(features,
                                                                                            encoded_labels,
                                                                                            test_size=0.15,
                                                                                            shuffle=True)

    if weighted == True:
        pos = np.count_nonzero(labels)
        neg = len(labels) - np.count_nonzero(labels)
        total = len(labels)
        initial_bias = np.log([pos / neg])

        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print(f'Weight for class 0: {weight_for_0:.2f}, class 1: {weight_for_1:.2f}')
    else:
        class_weight = None


    # train/validation
    training = model.fit(x=features_train, y=labels_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         shuffle=True,
                         verbose=0,
                         validation_data=(features_validation, labels_validation),
                         class_weight=class_weight)

    result = model.evaluate(features_test, encoded_labels_test, verbose=verbose)
    if verbose != 0:
        print("Test loss function: ", result[0])
        print("Test accuracy: ", result[1])
        print("Test precision: ", result[2])
        print("Test recall: ", result[3])
        print("Test F1 score: ", result[4])
        print()

    if show == True:
        plot_training(training)
        plt.show()

    # return model, (loss, accuracy, .., F1score)
    return model, result


def apply_neural_network(slt_features, labels,
                         nn_type="softmax",
                         optimizer='adam',
                         loss='categorical_crossentropy',
                         epochs=100,
                         batch_size=32,
                         verbose=0,
                         show=False,
                         weighted=False
                         ):

    slt_features = np.asarray(slt_features)

    labels = np.array(labels)
    encoded_labels = to_categorical(labels.reshape(-1, 1).astype(int))
    encoded_labels = np.array(encoded_labels)

    # visualize_labels_after_preprocessing(labels, encoded_labels)

    # Separate the test data
    features, features_test, encoded_labels, encoded_labels_test = train_test_split(slt_features,
                                                                                    encoded_labels, test_size=0.15,
                                                                                    shuffle=True)

    # Split the remaining data to train and validation
    features_train, features_validation, labels_train, labels_validation = train_test_split(features,
                                                                                            encoded_labels,
                                                                                            test_size=0.15,
                                                                                            shuffle=True)

    if weighted == True:
        pos = np.count_nonzero(labels)
        neg = len(labels) - np.count_nonzero(labels)
        total = len(labels)
        initial_bias = np.log([pos / neg])

        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print(f'Weight for class 0: {weight_for_0:.2f}, class 1: {weight_for_1:.2f}')
    else:
        class_weight = None

    model = choose_nn(nn_type, n_features=slt_features.shape[1], output_size=encoded_labels.shape[-1])

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', precision, recall, F1])

    # train/validation
    training = model.fit(x=features_train, y=labels_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         shuffle=True,
                         verbose=0,
                         validation_data=(features_validation, labels_validation),
                         class_weight=class_weight)

    result = model.evaluate(features_test, encoded_labels_test, verbose=verbose)
    if verbose != 0:
        print("Test loss function: ", result[0])
        print("Test accuracy: ", result[1])
        print("Test precision: ", result[2])
        print("Test recall: ", result[3])
        print("Test F1 score: ", result[4])
        print()

    if show == True:
        plot_training(training)
        plt.show()

    # return model, (loss, accuracy, .., F1score)
    return model, result


def apply_neural_network_20_times(slt_features, labels,
                                  nn_type="softmax",
                                  optimizer='adam',
                                  loss='categorical_crossentropy',
                                  epochs=100,
                                  batch_size=32,
                                  verbose=0,
                                  show=False,
                                  weighted=False):
    test_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1s = []

    for i in range(20):
        model, result = apply_neural_network(slt_features, labels,
                                             nn_type=nn_type,
                                             optimizer=optimizer,
                                             loss=loss,
                                             epochs=epochs,
                                             batch_size=batch_size,
                                             verbose=verbose,
                                             show=show,
                                             weighted=weighted)

        test_losses.append(result[0])
        test_accuracies.append(result[1])
        test_precisions.append(result[2])
        test_recalls.append(result[3])
        test_f1s.append(result[4])

    avg_test_loss = sum(test_losses) / len(test_losses)
    avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
    avg_test_precision = sum(test_precisions) / len(test_precisions)
    avg_test_recall = sum(test_recalls) / len(test_recalls)
    avg_test_f1 = sum(test_f1s) / len(test_f1s)

    print(f"Test loss function: {avg_test_loss}")
    print(f"Test accuracy: {avg_test_accuracy}")
    print(f"Test precision: {avg_test_precision}")
    print(f"Test recall: {avg_test_recall}")
    print(f"Test F1 score: {avg_test_f1}")

    # plt = plot_results(training, simNr, ord, ncyc)
    # plt.show()

    return avg_test_accuracy



# apply_neural_network2(simNr=15, ord=2, ncyc=1.5)
# apply_neural_network_best(simNr=33, ord=2, ncyc=1.5)
# apply_neural_network_best(simNr=8, ord=2, ncyc=1.5)
# apply_neural_network_best(simNr=15, ord=2, ncyc=1.5)
# apply_neural_network_best(simNr=84, ord=2, ncyc=1.5)
# apply_neural_network_best(simNr=63, ord=2, ncyc=1.5)
# apply_neural_network_best(simNr=75, ord=2, ncyc=1.5)
# # apply_neural_network_best(simNr=79, ord=2, ncyc=1.5)
