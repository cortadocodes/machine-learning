from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from machine_learning.datasets.visualisation import plot_decision_regions


iris = datasets.load_iris()
samples = iris.data[:, [2, 3]]
targets = iris.target

print('Class labels:', np.unique(targets))

training_samples, test_samples, training_targets, test_targets = train_test_split(
    samples,
    targets,
    test_size = 0.3,
    random_state = 1,
    stratify = targets
)

print('Label counts in targets:', np.bincount(targets))
print('Label counts in training_targets:', np.bincount(training_targets))
print('Label counts in test_targets:', np.bincount(test_targets))

scaler = StandardScaler()
scaler.fit(training_samples)
standardised_training_samples = scaler.transform(training_samples)
standardised_test_samples = scaler.transform(test_samples)

perceptron = Perceptron(eta0 = 0.1, random_state = 1)
perceptron.fit(standardised_training_samples, training_targets)
predictions = perceptron.predict(standardised_test_samples)

misclassified_samples = (predictions != test_targets).sum()
accuracy = 1 - misclassified_samples / len(training_samples)

print(f'Misclassified samples: {misclassified_samples}')
print(f'Accuracy: {round(accuracy, 2)}')

sklearn_accuracy = accuracy_score(test_targets, predictions)
print(f'Accuracy computed by sklearn: {sklearn_accuracy:.2f}')

another_sklearn_accuracy = perceptron.score(standardised_test_samples, test_targets)
print(f'Accuracy computed by sklearn Perceptron: {another_sklearn_accuracy:.2f}')

combined_standardised_samples = np.vstack((standardised_training_samples, standardised_test_samples))
combined_targets = np.hstack((training_targets, test_targets))

plot_decision_regions(
    samples = combined_standardised_samples,
    targets = combined_targets,
    classifier = perceptron,
    test_index = range(105, 150)
)
plt.xlabel('Standardised petal length')
plt.ylabel('Standardised petal width')
plt.legend(loc='upper left')
plt.show()
