import numpy as np


class Perceptron:
    """A linear classifier that mimics a single neuron, either "firing" or not depending on its net input.

    Type: supervised - binary classification

    Advantages:
    * Guaranteed to converge if training data is linearly separable

    Disadvantages:
    * Will only converge if the training data is linearly separable
    * Will only converge if the learning rate is low enough
    """

    def __init__(self, learning_rate = 0.01, number_of_training_iterations = 50, random_state_seed = 1):
        """Initialise a Perceptron. Note that for the learning rate to have an effect on the classification outcome, the
        weights must be initialised to non-zero values.

        :param float learning_rate: should be between 0 and 1
        :param int number_of_training_iterations:
        :param int random_state_seed: for random weight initialisation
        :var np.array weights: internal weights of the Perceptron
        :var list(int) errors_per_epoch: number of mis-classifications (updates) in each epoch
        """
        self.learning_rate = learning_rate
        self.number_of_training_iterations = number_of_training_iterations
        self.random_state = random_state_seed
        self.weights = np.array([])
        # Collect the number of errors per epoch in order to analyse the model during training.
        self.errors_per_epoch = []

    def fit(self, training_samples, true_labels):
        """Fit the Perceptron to the training data.

        :param np.array training_samples: samples in a matrix of shape (n_samples, n_features)
        :param np.array true_labels: target values in a vector of shape (n_samples)
        :return Perceptron:
        """
        random_number_generator = np.random.RandomState(self.random_state)
        self.weights = random_number_generator.normal(loc = 0.0, scale = 0.01, size = 1 + training_samples.shape[1])
        self.errors_per_epoch = []

        for _ in range(self.number_of_training_iterations):
            number_of_errors = 0

            for sample, true_label in zip(training_samples, true_labels):
                weight_update = self.learning_rate * (true_label - self.predict(sample))
                self.weights[0] += weight_update
                self.weights[1:] += weight_update * sample
                number_of_errors += int(weight_update != 0)

            self.errors_per_epoch.append(number_of_errors)

        return self

    def net_input(self, sample):
        """Calculate the net input of a sample into the Perceptron.

        :param np.array sample: shape (n_features,)
        :return float:
        """
        return self.weights[0] + np.dot(sample, self.weights[1:])

    def predict(self, sample):
        """Classify a sample according to the decision function (a Heaviside step function).

        :param np.array sample: shape (n_features,)
        :return np.array:
        """
        return np.where(self.net_input(sample) >= 0, 1, -1)
