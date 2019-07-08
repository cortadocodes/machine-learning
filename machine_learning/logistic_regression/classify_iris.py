from matplotlib import pyplot as plt
import numpy as np

from machine_learning.logistic_regression.logistic_regression import LogisticRegressionWithGradientDescent
from machine_learning.datasets.iris import samples
from machine_learning.datasets.iris import binary_labels
from machine_learning.datasets.visualisation import plot_decision_regions
from machine_learning.datasets.transformations.standardise import standardise_samples


zero_or_one_class_label_conditions = (binary_labels == 0) | (binary_labels == 1)
samples_subset = samples[zero_or_one_class_label_conditions]
targets_subset = binary_labels[zero_or_one_class_label_conditions]

standardised_samples = standardise_samples(samples_subset)

classifier = LogisticRegressionWithGradientDescent(
    learning_rate = 0.05,
    number_of_training_iterations = 1000,
    random_state_seed = 1
)

classifier.fit(standardised_samples, targets_subset)

plot_decision_regions(standardised_samples, targets_subset, classifier)
plt.xlabel('Petal length (standardised)')
plt.ylabel('Petal width (standardised)')
plt.legend(loc = 'upper left')
plt.show()
