import numpy as np


class LogisticRegressionWithGradientDescent:
    """An adaptive linear neuron implementation of logistic regression; an improvement on the perceptron; uses gradient
    descent of the logistic cost function (log-likelihood) to arrive at the global cost minimum.

    Type: supervised - binary classification.

    Notes:
    * Uses a sigmoid activation function in calculating errors
    """

    def __init__(self, learning_rate = 0.01, number_of_training_iterations = 50, random_state_seed = 1):
        """Initialise an adaptive linear neuron that uses gradient descent. Note that for the learning rate to
        have an effect on the classification outcome, the weights must be initialised to non-zero values.

        :param float learning_rate: should be between 0 and 1
        :param int number_of_training_iterations:
        :param int random_state_seed: for random weight initialisation
        :var np.array weights: internal weights of the AdaptiveLinearNeuronWithGradientDescent
        :var list(int) errors_per_epoch: number of mis-classifications (updates) in each epoch
        """
        self.learning_rate = learning_rate
        self.number_of_training_iterations = number_of_training_iterations
        self.random_state = random_state_seed
        self.weights = np.array([])
        self.cost = []

    def fit(self, samples, targets):
        """Fit the neuron to the training data.

        :param np.array samples: samples in a matrix of shape (n_samples, n_features)
        :param np.array targets: target values in a vector of shape (n_samples)
        :return AdaptiveLinearNeuronWithGradientDescent:
        """
        random_number_generator = np.random.RandomState(self.random_state)
        self.weights = random_number_generator.normal(loc = 0.0, scale = 0.01, size = 1 + samples.shape[1])
        self.cost = []

        for i in range(self.number_of_training_iterations):
            net_input = self.net_input(samples)
            errors = targets - self.activation(net_input)
            self.weights[0] += self.learning_rate * errors.sum()
            self.weights[1:] += self.learning_rate * samples.T.dot(errors)
            self.cost.append(self.calculate_cost(net_input, targets))

        return self

    def net_input(self, samples):
        """Calculate the net input of a sample into the neuron.

        :param np.array samples: shape (n_samples, n_features)
        :return float:
        """
        return self.weights[0] + np.dot(samples, self.weights[1:])

    def activation(self, net_input):
        """Calculate the sigmoid activation of the net input.

        :param np.array net_input: shape (n_samples)
        :return np.array:
        """
        return 1 / (1 + np.exp(-np.clip(net_input, -250, 250)))

    def calculate_cost(self, net_input, targets):
        """Calculate the value of the cost function.

        :param np.array errors: shape (n_samples)
        :return float:
        """
        return - (
            targets.dot(np.log(self.activation(net_input)))
            + (1 - targets).dot(np.log(1 - self.activation(net_input)))
        )

    def predict(self, samples):
        """Classify a sample according to the decision function (a Heaviside step function).

        :param np.array sample: shape (n_samples, n_features)
        :return np.array:
        """
        return np.where(self.activation(self.net_input(samples)) >= 0, 1, 0)
