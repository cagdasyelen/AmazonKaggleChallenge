import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


class LogRes(object):
    def __init__(self, train_file, test_file, out_file):
        self.train_file = train_file
        self.test_file = test_file
        self.out_file = out_file

        self.data = self.__read_data(train_file)
        self.X_train_raw = None
        self.X_train_sparse = None
        self.y = None

        self.keymap = None
        self.model = None

        self.test_data = self.__read_data(test_file)
        self.X_test_raw = None
        self.X_test_sparse = None

        self.__extract_xy_train()
        self.__extract_xy_test()

    def __read_data(self, filename):
        return pd.read_csv(filename)

    def __extract_xy_train(self):
        """Extracts input matrix (X_train) and label vector (y) from csv parsed
        labelled data.
        """
        self.X_train_raw = self.data[['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1',
                                      'ROLE_ROLLUP_2', 'ROLE_DEPTNAME',
                                      'ROLE_TITLE', 'ROLE_FAMILY_DESC',
                                      'ROLE_FAMILY', 'ROLE_CODE'
                                      ]].as_matrix()
        n = self.X_train_raw.shape[0]
        self.X_train_raw = np.hstack((self.X_train_raw,
                                      np.ones(shape=(n, 1))))
        self.X_train_sparse = self.__one_hot_encoder(self.X_train_raw, True)
        self.y = self.data['ACTION'].as_matrix().reshape(-1, 1).ravel()

    def __extract_xy_test(self):
        """Extracts input matrix (X_train) from csv parsed test data.
        """
        self.X_test_raw = self.test_data[['RESOURCE', 'MGR_ID',
                                          'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2',
                                          'ROLE_DEPTNAME', 'ROLE_TITLE',
                                          'ROLE_FAMILY_DESC', 'ROLE_FAMILY',
                                          'ROLE_CODE']].as_matrix()
        n = self.X_test_raw.shape[0]
        self.X_test_raw = np.hstack((self.X_test_raw,
                                     np.ones(shape=(n, 1))))
        self.X_test_sparse = self.__one_hot_encoder(self.X_test_raw, False)

    def __one_hot_encoder(self, raw_data, train):
        """Given a matrix with categorical columns, returns a sparse matrix
        with one-hot encoded columns. Also sets the keymap for categorical to
        index mapping. The keymap is set only during training.

        To handle OOV categories during testing, the first occurrence of each
        category in a column during training is treated as OOV (all features
        for that category in that particular training eg are set to 0)

        Args:
            raw_data (matrix): matrix of raw categorical data
            train (bool): set of True during training

        Returns:
            sparse csr: sparse one-hot encoded matrix
        """
        n = raw_data.shape[0]

        if self.keymap is None:
            self.keymap = []
            for col in raw_data.T:
                unique = set(col)
                self.keymap.append(dict((v, i) for i, v in enumerate(unique)))

        sparse_data = []
        # seen = []   # to keep track of first occurrence of each ID in a column
        for c, col in enumerate(raw_data.T):
            col_matrix = sparse.lil_matrix((n, len(self.keymap[c])))
            # if train:   # treating first occ as OOV only during training
                # seen.append(set())

            for r, v in enumerate(col):
                # if train and (v not in seen[-1]):  # first occurrence of v
                    # seen[-1].add(v)  # add v to seen set. Keep the feat vec 0s.
                if v in self.keymap[c]:
                    col_matrix[r, self.keymap[c][v]] = 1

            sparse_data.append(col_matrix)

        return sparse.hstack(sparse_data).tocsr()

    def __cross_validation(self, X, y):
        """Performs 10-fold cross validation on the supplied feature matrix
        and labels

        Args:
            X (2d): Input feature matrix
            y (1d): Labels
        """
        scores = cross_val_score(self.model,
                                 X, y, scoring='accuracy',
                                 cv=10, n_jobs=4)
        print scores
        print scores.mean()

    def __train_and_predict(self, X_train, y, X_test):
        """Trains the model on supplied training matrix and labels, and write
        prediction probabilities on the supplied test matrix to csv.

        Args:
            X_train (2d): Training feature matrix.
            y (1d): Label.
            X_test (2d): Test feature matrix.
        """
        self.model.fit(X_train, y)
        print "Training = %f" % self.model.score(X_train, y)
        prediction_probs = self.model.predict_proba(X_test)
        self.__write_csv(prediction_probs)

    def __write_csv(self, prediction_probs):
        n = self.X_test_sparse.shape[0]
        d = {'Id': pd.Series([i for i in xrange(1, n + 1)]),
             'Action': pd.Series(prediction_probs[:, 1])}
        df = pd.DataFrame(d)
        df = df[['Id', 'Action']]
        df.to_csv(self.out_file, sep=',', encoding='utf-8',
                  index=False)

    def logistic_regression(self):
        self.model = LogisticRegression(class_weight='balanced')
        self.__cross_validation(self.X_train_sparse, self.y)
        # self.__train_and_predict(self.X_train_sparse,
        #                          self.y, self.X_test_sparse)


def main(train_file, test_file, out_file):
    lr = LogRes(train_file, test_file, out_file)
    lr.logistic_regression()


if __name__ == '__main__':
    train_file = "./data/train.csv"
    test_file = "./data/test.csv"
    out_file = "./data/ensemble_results/lr_unbalanced_sparse.csv"
    main(train_file, test_file, out_file)
