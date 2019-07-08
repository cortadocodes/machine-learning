import numpy as np


def standardise_samples(samples):
    """ Use the standardisation method of feature scaling to give all values of a feature in a sample the properties of
    a standardised normal distribution.

    :param np.array samples: shape(n_samples, n_features
    :return np.array:
    """
    standardised_samples = np.copy(samples)

    standardised_samples[:, 0] = (
        standardised_samples[:, 0] - standardised_samples[:, 0].mean()
    ) / standardised_samples[:, 0].std()

    standardised_samples[:, 1] = (
        standardised_samples[:, 1] - standardised_samples[:, 1].mean()
    ) / standardised_samples[:, 1].std()

    return standardised_samples
