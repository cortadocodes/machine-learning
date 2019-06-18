import numpy as np


class AdaptiveLinearNeuron:
    """An adaptive linear neuron (Adaline); an improvement on the perceptron.

    Type: supervised - binary classification
    """

    def __init__(self, learning_rate=0.01, number_of_training_iterations=50, random_state_seed=1):
        """Initialise a AdaptiveLinearNeuron. Note that for the learning rate to have an effect on the classification
        outcome, the weights must be initialised to non-zero values.

        :param float learning_rate: should be between 0 and 1
        :param int number_of_training_iterations:
        :param int random_state_seed: for random weight initialisation
        :var np.array weights: internal weights of the AdaptiveLinearNeuron
        :var list(int) errors_per_epoch: number of mis-classifications (updates) in each epoch
        """
        self.learning_rate = learning_rate
        self.number_of_training_iterations = number_of_training_iterations
        self.random_state = random_state_seed
        self.weights = np.array([])
        self.cost = []

    def fit(self, samples, true_labels):
        """Fit the AdaptiveLinearNeuron to the training data.

        :param np.array samples: samples in a matrix of shape (n_samples, n_features)
        :param np.array true_labels: target values in a vector of shape (n_samples)
        :return AdaptiveLinearNeuron:
        """
        random_number_generator = np.random.RandomState(self.random_state)
        self.weights = random_number_generator.normal(loc=0.0, scale=0.01, size=1 + samples.shape[1])
        self.cost = []

        for i in range(self.number_of_training_iterations):
            net_input = self.net_input(samples)
            output = self.activation(net_input)
            errors = true_labels - output
            self.weights[0] += self.learning_rate * errors.sum()
            self.weights[1:] += self.learning_rate * samples.T.dot(errors)
            self.cost.append(self.cost(errors))

        return self

    def net_input(self, samples):
        """Calculate the net input of a sample into the AdaptiveLinearNeuron.

        :param np.array samples: shape (n_samples, n_features)
        :return float:
        """
        return np.dot(samples, self.weights[1:]) + self.weights[0]

    def activation(self, samples):
        """Calculate the linear activation.

        :param np.array samples: shape (n_samples, n_features)
        :return np.array:
        """
        return samples

    def cost(self, errors):
        """Calculate the value of the cost function.

        :param np.array errors: shape (n_samples)
        :return float:
        """
        return (errors**2).sum() / 2

    def predict(self, samples):
        """Classify a sample according to the decision function (a Heaviside step function).

        :param np.array sample: shape (n_samples, n_features)
        :return np.array:
        """
        return np.where(self.activation(self.net_input(samples)) >= 0, 1, -1)
