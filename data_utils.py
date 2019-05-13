from keras import datasets
import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.backend as K
from sklearn import model_selection, metrics
from sklearn.metrics import classification_report
import math


# data evalutate function
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())

# data pre-processing function
# standardization of data
def data_std(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))

    X_train = (X_train - mean)/(std + 1e-7)
    X_test = (X_test - mean)/(std + 1e-7)

    return X_train, X_test

# plot history for accuracy
def plot_dice(history, title = None):
    fig = plt.figure()
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['dice_coef'])
    plt.plot(history['val_dice_coef'])
    if title is not None:
        plt.title(title)

    plt.ylabel('Dice_coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc=0)

# plot history for accuracy
def plot_acc(history, title = None):
    fig = plt.figure()
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)

    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc=0)



# plot history for loss
def plot_loss(history, title = None):
    fig = plt.figure()
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)

    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc=0)
