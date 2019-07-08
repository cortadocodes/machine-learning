from matplotlib import pyplot as plt

from machine_learning.adaptive_linear_neuron.adaptive_linear_neuron_with_gradient_descent import \
    AdaptiveLinearNeuronWithGradientDescent
from machine_learning.datasets.iris import samples
from machine_learning.datasets.iris import binary_labels
from machine_learning.perceptron.classify_iris import plot_decision_regions
from machine_learning.datasets.transformations import standardise_samples


adaline = AdaptiveLinearNeuronWithGradientDescent(learning_rate = 0.01, number_of_training_iterations = 15)
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
