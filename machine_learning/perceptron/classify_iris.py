from matplotlib import pyplot as plt

from machine_learning.datasets.iris import samples
from machine_learning.datasets.iris import binary_labels
from machine_learning.datasets.visualisation import plot_decision_regions
from machine_learning.perceptron import Perceptron


def plot_data(samples):
    """ Plot the Setosa and Versicolor sepal and petal length data.

    :param pd.DataFrame samples:
    :return None:
    """
    plt.scatter(samples[:50, 0], samples[:50, 1], color = 'red', marker = 'o', label = 'setosa')
    plt.scatter(samples[50:100, 0], samples[50:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
    plt.xlabel('Sepal length (cm)')
    plt.ylabel('Petal length (cm)')
    plt.legend(loc = 'upper left')
    plt.show()


def plot_errors_per_epoch(perceptron):
    """ Plot the errors per epoch of the Perceptron as it is trained.

    :param perceptron.Perceptron perceptron:
    :return None:
    """
    epochs = range(len(perceptron.errors_per_epoch))
    plt.plot(epochs, perceptron.errors_per_epoch, marker = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('Errors per epoch')
    plt.show()


if __name__ == '__main__':
    plot_data(samples)

    perceptron = Perceptron(learning_rate = 0.1, number_of_training_iterations = 10)
    perceptron.fit(samples, binary_labels)

    plot_errors_per_epoch(perceptron)
    plot_decision_regions(samples, binary_labels, perceptron)
