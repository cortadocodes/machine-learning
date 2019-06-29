import numpy as np


class AdaptiveLinearNeuronWithStochasticGradientDescent:
    """An adaptive linear neuron (Adaline); an improvement on the perceptron. This Adaline uses stocharstic gradient
    descent to arrive at the global cost minimum more quickly than via batch gradient descent.

    Type: supervised - binary classification.

    Notes:
    * Uses a linear activation function in calculating errors
    * Uses a sum-squared-error (SSE) function as the cost function
    """

    def __init__(self, learning_rate = 0.01, number_of_training_iterations = 50, random_state_seed = 1, shuffle = True):
        """Initialise an adaptive linear neuron that uses gradient descent. Note that for the learning rate to
        have an effect on the classification outcome, the weights must be initialised to non-zero values.

        :param float learning_rate: should be between 0 and 1
        :param int number_of_training_iterations:
        :param int random_state_seed: for random weight initialisation
        :param bool shuffle:
        """
        self.learning_rate = learning_rate
        self.number_of_training_iterations = number_of_training_iterations
        self.random_state = random_state_seed
        self.shuffle = shuffle
        self.weights = np.array([])
        self.weights_initialised = False
        self.cost = []

    def fit(self, samples, labels):
        """Fit the neuron to the training data.

        :param np.array samples: samples in a matrix of shape (n_samples, n_features)
        :param np.array labels: target values in a vector of shape (n_samples)
        :return AdaptiveLinearNeuronWithStochasticGradientDescent:
        """
        self._initialise_weights(samples.shape[1])
        self.cost = []

        for i in range(self.number_of_training_iterations):

            if self.shuffle:
                samples, labels = self._shuffle(samples, labels)

            average_cost = np.mean([self._update_weights(samples, label) for sample, label in zip(samples, labels)])
            self.cost.append(average_cost)

        return self

    def incremental_fit(self, samples, labels):
        """Fit the model without reinitialising the weights (useful for online or incremental learning).

        :param np.array samples: shape (n_samples, n_features)
        :param np.array labels: shape (n_samples)
        :return AdaptiveLinearNeuronWithStochasticGradientDescent:
        """
        if not self.weights_initialised:
            self._initialise_weights(samples.shape[1])

        if labels.ravel().shape[0] > 1:
            for sample, label in zip(samples, labels):
                self._update_weights(sample, label)

        else:
            self._update_weights(samples, labels)

        return self

    def net_input(self, samples):
        """Calculate the net input of a sample into the neuron.

        :param np.array samples: shape (n_samples, n_features)
        :return float:
        """
        return self.weights[0] + np.dot(samples, self.weights[1:])

    @staticmethod
    def activation(net_input):
        """Calculate the linear activation of the net input.

        :param np.array net_input: shape (n_samples)
        :return np.array:
        """
        return net_input

    @staticmethod
    def calculate_cost(errors):
        """Calculate the value of the cost function.

        :param np.array errors: shape (n_samples)
        :return float:
        """
        return (errors**2).sum() / 2

    def predict(self, samples):
        """Classify a sample according to the decision function (a Heaviside step function).

        :param np.array samples: shape (n_samples, n_features)
        :return np.array:
        """
        return np.where(self.activation(self.net_input(samples)) >= 0, 1, -1)

    def _initialise_weights(self, length):
        """Initialise weights to small random numbers.

        :param int length:
        :return None:
        """
        self.random_number_generator = np.random.RandomState(self.random_state)
        self.weights = self.random_number_generator.normal(loc = 0.0, scale = 0.01, size = 1 + length)
        self.weights_initialised = True

    def _update_weights(self, sample, label):
        """Update the weights according by training on one sample.

        :param np.array sample: shape (n_features)
        :param int label:
        :return float:
        """
        net_input = self.net_input(sample)
        error = label - self.activation(net_input)
        self.weights[0] += self.learning_rate * error
        self.weights[1:] += self.learning_rate * sample.dot(error)
        cost = 0.5 * error**2
        return cost

    def _shuffle(self, samples, labels):
        """Shuffle the training data.

        :param np.array samples: shape(n_samples, n_features)
        :param np.array labels: shape (n_samples)
        :return tuple(np.array):
        """
        indices_permutation = self.random_number_generator.permutation(len(labels))
        return samples[indices_permutation], labels[indices_permutation]
