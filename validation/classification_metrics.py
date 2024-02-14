from tensorflow.python.keras.backend import sum
from tensorflow.python.keras.backend import round as tfround
from tensorflow.python.keras.backend import clip
from tensorflow.python.keras.backend import epsilon

# functie care defineste modul de calcul al scorului F1
def F1(y_true, y_pred):
    true_positives = sum(tfround(clip(y_true * y_pred, 0, 1)))
    predicted_positives = sum(tfround(clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon())

    possible_positives = sum(tfround(clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon())

    return 2 * ((precision * recall) / (precision + recall + epsilon()))

def precision(y_true, y_pred):
    true_positives = sum(tfround(clip(y_true * y_pred, 0, 1)))
    predicted_positives = sum(tfround(clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon())

    return precision

def recall(y_true, y_pred):
    true_positives = sum(tfround(clip(y_true * y_pred, 0, 1)))

    possible_positives = sum(tfround(clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon())

    return recall
