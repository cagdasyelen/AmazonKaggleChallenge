import numpy as np
import pandas as pd
from itertools import combinations


def extract_xy_train(data, group, degrees=[]):
    """Extracts feature matrix (X_train) and label vector (y) from csv
    parsed labelled training data (pd df). Optionally adds feature groupings
    to the feature matrix.

    Args:
        data (pandas df): Dataframe parsed from csv file
        group (bool): True if feature grouping is desired.
        degrees (list(int), optional): List of degrees of feature
            combinations to be added, eg [2] (groups of 2 only) or [2, 3]
            (groups of 2 and 3 both)
    """
    X_train_raw = data.ix[:, :].values
    if group and (len(degrees) > 0 and X_train_raw.shape[1] >= min(degrees)):
        X_train_raw = add_groupings(X_train_raw, degrees)

    y = data['ACTION'].as_matrix().reshape(-1, 1).ravel()
    return X_train_raw, y


def extract_xy_test(test_data, group, degrees=[]):
    """Extracts feature matrix (X_test) and label vector (y) from csv
    parsed unlabelled test data (pd df). Only the specified columns are used.
    Optionally adds feature groupings to the feature matrix.

    Args:
        test_data (pandas df): Dataframe parsed from csv file.
        group (bool): True if feature grouping is desired.
        degrees (list(int), optional): List of degrees of feature
            combinations to be added, eg [2] (groups of 2 only) or [2, 3]
            (groups of 2 and 3 both)
    """
    X_test_raw = test_data.ix[:, :].values
    if group and (len(degrees) > 0 and X_test_raw.shape[1] >= min(degrees)):
        X_test_raw = add_groupings(X_test_raw, degrees)
    return X_test_raw


def add_groupings(data, degrees):
    """Constructs new features from groups of existing features, and adds
    it to the supplied feature matrix.

    Args:
        data (np.array): Input feature matrix.
        degrees (list(int)): List of degrees of feature combinations
            to be added.
    """
    grouped_data = [data]
    for d in degrees:
        grouped_data.append(group_data(data, d))
    print ""
    return np.hstack([g for g in grouped_data])


def group_data(data, degree):
    new_data = []
    m, n = data.shape
    for indices in combinations(range(n), degree):
        print indices,
        new_data.append([hash(tuple(v)) for v in data[:, indices]])
    return np.array(new_data).T

if __name__ == '__main__':
    extract_xy_test(pd.DataFrame(), True, [2, 3])
