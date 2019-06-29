from matplotlib import pyplot as plt
import numpy as np

from machine_learning.adaptive_linear_neuron.adaptive_linear_neuron_with_gradient_descent import \
    AdaptiveLinearNeuronWithGradientDescent
from machine_learning.datasets.iris import samples
from machine_learning.datasets.iris import binary_labels


adaline_1 = AdaptiveLinearNeuronWithGradientDescent(learning_rate = 0.01, number_of_training_iterations = 10)
adaline_1.fit(samples, binary_labels)

adaline_2 = AdaptiveLinearNeuronWithGradientDescent(learning_rate = 0.0001, number_of_training_iterations = 10)
adaline_2.fit(samples, binary_labels)

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))

ax[0].plot(
    range(1, len(adaline_1.cost) + 1),
    np.log10(adaline_1.cost),
    marker = 'o'
)
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(sum-squared-error)')
ax[0].set_title('Adaptive linear neuron - learning rate 0.01')

ax[1].plot(
    range(1, len(adaline_2.cost) + 1),
    adaline_2.cost,
    marker = 'o'
)
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('sum-squared-error')
ax[1].set_title('Adaptive linear neuron - learning rate 0.0001')

plt.show()
