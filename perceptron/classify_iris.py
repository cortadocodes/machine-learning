from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from perceptron import Perceptron


IRIS_DATASET_LOCATION = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


def extract_and_convert_labels(dataframe, number_of_samples=100, label_column_index=4):
    """ Extract and convert the English binary_labels to binary binary_labels (-1 for Iris-setosa; 1 otherwise)

    :param pd.DataFrame dataframe:
    :param int number_of_samples:
    :param int label_column_index:
    :return np.array:
    """
    labels = dataframe.iloc[0:number_of_samples, label_column_index].values
    return np.where(labels == 'Iris-setosa', -1, 1)


def extract_samples(dataframe, number_of_samples=100, feature_column_indices=None):
    """ Extract samples.

    :param pd.DataFrame dataframe:
    :param int number_of_samples:
    :param list(int) feature_column_indices:
    :return pd.DataFrame:
    """
    feature_column_indices = feature_column_indices or [0, 2]
    return dataframe.iloc[0:number_of_samples, feature_column_indices].values


def plot_data(samples):
    """ Plot the Setosa and Versicolor sepal and petal length data.

    :param pd.DataFrame samples:
    :return None:
    """
    plt.scatter(samples[:50, 0], samples[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(samples[50:100, 0], samples[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('Sepal length (cm)')
    plt.ylabel('Petal length (cm)')
    plt.legend(loc='upper left')
    plt.show()


def plot_errors_per_epoch(perceptron):
    """ Plot the errors per epoch of the Perceptron as it is trained.

    :param perceptron.Perceptron perceptron:
    :return None:
    """
    epochs = range(len(perceptron.errors_per_epoch))
    plt.plot(epochs, perceptron.errors_per_epoch, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Errors per epoch')
    plt.show()


def plot_decision_regions(samples, labels, classifier, resolution=0.02):
    """ Plot the training data with the decision boundary of the trained classifier.

    :param pd.DataFrame samples:
    :param np.array labels:
    :param mixed classifier:
    :param float resolution:
    :return None:
    """
    # Set up markers and colour map.
    markers = ('s', 'x', 'o', '^', 'v')
    colours = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    possible_classification_labels = np.unique(labels)
    colour_map = ListedColormap(colours[:len(possible_classification_labels)])

    # Set range of features to view
    feature_1_min = samples[:, 0].min() - 1
    feature_1_max = samples[:, 0].max() + 1
    feature_2_min = samples[:, 1].min() - 1
    feature_2_max = samples[:, 1].max() + 1

    # Set up grid filling this range.
    feature_1_range = np.arange(feature_1_min, feature_1_max, resolution)
    feature_2_range = np.arange(feature_2_min, feature_2_max, resolution)
    xx1, xx2 = np.meshgrid(feature_1_range, feature_2_range)

    predictions = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape)

    # Create contour plot of classification.
    plt.contourf(xx1, xx2, predictions, alpha=0.3, cmap=colour_map)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class samples
    for index, classification in enumerate(possible_classification_labels):
        plt.scatter(
            x=samples[labels == classification, 0],
            y=samples[labels == classification, 1],
            alpha=0.8,
            c=colours[index],
            marker=markers[index],
            label=classification,
            edgecolor='black'
        )

    plt.xlabel('Sepal length (cm)')
    plt.ylabel('Petal length (cm)')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    iris_dataset = pd.read_csv(IRIS_DATASET_LOCATION, header=None)
    samples = extract_samples(iris_dataset)
    binary_labels = extract_and_convert_labels(iris_dataset)
    plot_data(samples)

    perceptron = Perceptron(learning_rate=0.1, number_of_training_iterations=10)
    perceptron.fit(samples, binary_labels)

    plot_errors_per_epoch(perceptron)
    plot_decision_regions(samples, binary_labels, perceptron)
