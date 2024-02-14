import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.backend import round
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model


def build_cnn(data_size, n_features):
    model = tf.keras.models.Sequential()
    # input layer
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(data_size, n_features)))

    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='relu'))

    # output layer
    model.add(tf.keras.layers.Dense(1, activation='softmax'))

    return model


# functie care construieste reteaua neuronala
def build_neural_network(n_features, output_size, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    inputLayer = Input(name="input", shape=(n_features,))
    hiddenLayer = Dense(name="hidden", units=int(round((n_features + 1) / 2)), activation='relu')(inputLayer)
    outputLayer = Dense(name="output", units=output_size, activation='sigmoid', bias_initializer=output_bias)(
        hiddenLayer)

    model = Model(inputs=inputLayer, outputs=outputLayer, name="neuralNetwork")
    # model.summary()

    return model


def build_neural_network_deep(n_features, output_size):
    inputLayer = Input(name="input", shape=(n_features,))
    hiddenLayer1 = Dense(name="hidden1", units=int(round((n_features + 1) / 2)), activation='relu')(inputLayer)
    hiddenLayer2 = Dense(name="hidden2", units=int(round((n_features + 1) / 4)), activation='relu')(hiddenLayer1)
    hiddenLayer3 = Dense(name="hidden3", units=int(round((n_features + 1) / 8)), activation='relu')(hiddenLayer2)
    outputLayer = Dense(name="output", units=output_size, activation='sigmoid')(hiddenLayer3)

    model = Model(inputs=inputLayer, outputs=outputLayer, name="neuralNetwork")
    # model.summary()

    return model


def build_neural_network_softmax(n_features, output_size, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    inputLayer = Input(name="input", shape=(n_features,))
    hiddenLayer = Dense(name="hidden1", units=int(round((n_features + 1) / 2)), activation='relu')(inputLayer)
    # hiddenLayer = Dense(name="hidden2", units=int(round((n_features + 1) / 4)), activation='relu')(hiddenLayer)
    # hiddenLayer = Dense(name="hidden3", units=int(round((n_features + 1) / 8)), activation='relu')(hiddenLayer)
    outputLayer = Dense(name="output", units=output_size, activation='softmax', bias_initializer=output_bias)(hiddenLayer)

    model = Model(inputs=inputLayer, outputs=outputLayer, name="neuralNetwork")
    # model.summary()

    return model


def choose_nn(type, n_features, output_size):
    if type == "sigmoid":
        return build_neural_network(n_features, output_size)
    if type == "sigmoid_deep":
        return build_neural_network_deep(n_features, output_size)
    elif type == "softmax":
        return build_neural_network_softmax(n_features, output_size)
