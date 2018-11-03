import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # Set up markers and colour map
    markers = ('s', 'x', 'o', '^', 'v')
    colours = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    unique_classifications = np.unique(y)
    colour_map = ListedColormap(colours[:len(unique_classifications)])

    # Set range of x1, x2 to view
    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1
    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1

    # Set up grid filling this range
    x1_range = np.arange(x1_min, x1_max, resolution)
    x2_range = np.arange(x2_min, x2_max, resolution)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)

    # Predict and reshape
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)

    # Create contour plot of classification
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=colour_map)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class samples
    for idx, classification in enumerate(unique_classifications):
        plt.scatter(
            x=X[y == classification, 0],
            y=X[y == classification, 1],
            alpha=0.8,
            c=colours[idx],
            marker=markers[idx],
            label=classification,
            edgecolor='black'
        )

    plt.xlabel('sepal length (cm)')
    plt.ylabel('petal length (cm)')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(LOCATION, header=None)
    X = extract_features(df)
    y = extract_labels(df)
    plot_data(X)

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    plot_errors(ppn)
    plot_decision_regions(X, y, ppn)
