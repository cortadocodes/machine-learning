from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def plot_decision_regions(samples, targets, classifier, resolution = 0.02, test_index = None):
    """ Plot the training data with the decision boundary of the trained classifier.

    :param pd.DataFrame samples:
    :param np.array targets:
    :param mixed classifier:
    :param float resolution:
    :return None:
    """
    # Set up markers and colour map.
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    possible_classification_labels = np.unique(targets)
    colour_map = ListedColormap(colors[:len(possible_classification_labels)])

    # Set range of features to view
    feature_1_min = samples[:, 0].min() - 1
    feature_1_max = samples[:, 0].max() + 1
    feature_2_min = samples[:, 1].min() - 1
    feature_2_max = samples[:, 1].max() + 1

    # Set up grid filling this range.
    feature_1_range = np.arange(feature_1_min, feature_1_max, resolution)
    feature_2_range = np.arange(feature_2_min, feature_2_max, resolution)
    xx1, xx2 = np.meshgrid(feature_1_range, feature_2_range)

    predictions = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape)

    # Create contour plot of classification.
    plt.contourf(xx1, xx2, predictions, alpha=0.3, cmap=colour_map)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class samples
    for index, classification in enumerate(possible_classification_labels):
        plt.scatter(
            x = samples[targets == classification, 0],
            y = samples[targets == classification, 1],
            alpha = 0.8,
            c = colors[index],
            marker = markers[index],
            label = classification,
            edgecolor = 'black'
        )

    # Highlight test samples
    if test_index:
        test_samples, test_targets = samples[test_index, :], targets[test_index]

        plt.scatter(
            test_samples[:, 0],
            test_samples[:, 1],
            c = '',
            edgecolor = 'black',
            alpha = 1.0,
            linewidth = 1,
            marker = 'o',
            s = 100,
            label = 'Test set'
        )
