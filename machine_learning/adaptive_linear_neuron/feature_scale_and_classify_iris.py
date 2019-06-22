from matplotlib import pyplot as plt
import numpy as np

from machine_learning.adaptive_linear_neuron.adaptive_linear_neuron import AdaptiveLinearNeuron
from machine_learning.datasets.iris import samples
from machine_learning.datasets.iris import binary_labels
from machine_learning.perceptron.classify_iris import plot_decision_regions


def standardise_samples(samples):
    """ Use the standardisation method of feature scaling to give all values of a feature in a sample the properties of
    a standardised normal distribution.

    :param np.array samples: shape(n_samples, n_features
    :return np.array:
    """
    standardised_samples = np.copy(samples)

    standardised_samples[:, 0] = (
        standardised_samples[:, 0] - standardised_samples[:, 0].mean()
    ) / standardised_samples[:, 0].std()

    standardised_samples[:, 1] = (
        standardised_samples[:, 1] - standardised_samples[:, 1].mean()
    ) / standardised_samples[:, 1].std()

    return standardised_samples


adaline = AdaptiveLinearNeuron(learning_rate=0.01, number_of_training_iterations=15)
standardised_samples = standardise_samples(samples)
adaline.fit(standardised_samples, binary_labels)

plot_decision_regions(standardised_samples, binary_labels, classifier = adaline)

plt.plot(
    range(1, len(adaline.cost) + 1),
    adaline.cost,
    marker = 'o'
)
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error (SSE)')
plt.show()