import numpy as np


class Perceptron:
    """Perceptron classifier.

    A linear classifier that mimics a single neuron, either "firing" or not depending on its inputs.

    Type: supervised - binary classification

    Advantages:
    * Guaranteed to converge if training data is linearly separable

    Disadvantages:
    * Will only converge if the training data is linearly separable

    :param float eta: learning rate (between 0 and 1)
    :param int n_iter: training iterations
    :param int random_state: random number generator seed for random weight initialisation
    :param np.array weights_: weights after fitting
    :param list errors_: number of mis-classifications (updates) in each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

        self.weights_ = np.array([])
        self.errors_ = []

    def fit(self, X, y):
        """Fit training data.
        :param np.array X: training vectors in a matrix of size n_samples by n_features
        :param np.array y: target values in a vector

        :return Perceptron:
        """
        random_number_generator = np.random.RandomState(self.random_state)
        self.weights_ = random_number_generator.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.weights_[1:] += update * xi
                self.weights_[0] += update
                errors += int(update != 0)
            self.errors_.append(errors)

        return self

    def net_input(self, X):
        """Calculate net input.

        :param np.array X: input values to net

        :return np.array: net input
        """
        net_input = np.dot(X, self.weights_[1:]) + self.weights_[0]
        return net_input

    def predict(self, X):
        """Predict class label after unit step.

        :param np.array X: input values to predict on
        """
        predictions = np.where(self.net_input(X) >= 0, 1, -1)
        return predictions
