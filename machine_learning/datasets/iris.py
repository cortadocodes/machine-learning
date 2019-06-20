import numpy as np
import pandas as pd


IRIS_DATASET_LOCATION = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


def extract_and_convert_labels(dataframe, number_of_samples=100, label_column_index=4):
    """ Extract and convert the English binary_labels to binary binary_labels (-1 for Iris-setosa; 1 otherwise)

    :param pd.DataFrame dataframe:
    :param int number_of_samples:
    :param int label_column_index:
    :return np.array:
    """
    labels = dataframe.iloc[0:number_of_samples, label_column_index].values
    return np.where(labels == 'Iris-setosa', -1, 1)


def extract_samples(dataframe, number_of_samples=100, feature_column_indices=None):
    """ Extract samples.

    :param pd.DataFrame dataframe:
    :param int number_of_samples:
    :param list(int) feature_column_indices:
    :return pd.DataFrame:
    """
    feature_column_indices = feature_column_indices or [0, 2]
    return dataframe.iloc[0:number_of_samples, feature_column_indices].values


iris_dataset = pd.read_csv(IRIS_DATASET_LOCATION, header=None)
samples = extract_samples(iris_dataset)
binary_labels = extract_and_convert_labels(iris_dataset)