import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from perceptron import Perceptron


LOCATION = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


def extract_labels(df):
    labels = df.iloc[0:100, 4].values
    labels = np.where(labels == 'Iris-setosa', -1, 1)
    return labels


def extract_features(df):
    features = df.iloc[0:100, [0, 2]].values
    return features


def plot_data(X):
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('petal length (cm)')
    plt.legend(loc='upper left')
    plt.show()


def plot_errors(ppn):
    epochs = range(1, len(ppn.errors_) + 1)
    plt.plot(epochs, ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Errors/number of updates')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(LOCATION, header=None)
    X = extract_features(df)
    y = extract_labels(df)
    plot_data(X)

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plot_errors(ppn)
