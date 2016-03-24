import xgboost as xgb
from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix,hstack
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import *

class Xgboost:
    def __init__(self,K,dirName):
        self.K = K
        self.load_data_from_dir(dirName)
        '''
        best_maxdepth, best_nestimators, best_colsamplebytree = \
                            self.get_best_params()
        '''
        self.xgb_model = xgb.XGBClassifier()
        self.classifier = self.cross_validator()
        '''
        self.classifier = xgb.XGBClassifier(max_depth=best_maxdepth, \
                                        n_estimators=best_nestimators, \
                                        colsample_bytree=best_colsamplebytree)
        '''
        self.train()
        self.predict()

    def cross_validator(self):
        parameters = {
            'objective':['binary:logistic'],
            'max_depth' :[6,8,10,12],
            'n_estimators' : [300,500,800,1000,1400],
            'colsample_bytree' : [0.5,0.7,1]
        }
        clf = GridSearchCV(self.xgb_model,parameters, n_jobs=4,\
            cv=StratifiedKFold(self.train_Y, n_folds=5, shuffle=True), \
            verbose=2, refit=True)

        return clf



    def load_data_from_dir(self,dirName):
        train_file = './'+ dirName + '/train.csv'
        test_file = './' + dirName + '/test.csv'
        df_train = pd.read_csv(train_file, header=0)
        df_test = pd.read_csv(test_file, header=0)
        # Get train_X, train_y, test_X
        train = df_train.values
        self.train_X = train[0::,1::]
        self.train_Y = train[0::,0]
        self.test_X = df_test.values[0::,1::]
        # Engineer the features in both train and test data
        self.train_X, self.test_X = \
                            self.engineer_all_features(self.train_X,self.test_X)
        self.test_ID = df_test.values[0::,0]

    def engineer_all_features(self, train, test):
        num_features = train.shape[1]
        train_new = []
        test_new = []

        for j in np.arange(0,num_features):
            train_sparse,test_sparse = self.engineer_feature(\
                                            train[0::,j], test[0::,j])

            train_new.append(train_sparse)
            test_new.append(test_sparse)

        # append the sparse matrix
        train_new = hstack(train_new).tocsr()
        test_new = hstack(test_new).tocsr()
        return (train_new, test_new)

    def engineer_feature(self, train_col, test_col):
        new_feature_col_train = np.zeros(train_col.shape)
        new_feature_col_test = np.zeros(test_col.shape)
        dictionary = {}
        num_keys_found = 0

        # Create the dictionary
        for row in train_col:
            if not dictionary.has_key(row):
                dictionary[row] = num_keys_found
                num_keys_found += 1

        # Update the columns for train
        for i in np.arange(0, train_col.size):
            new_feature_col_train[i] = dictionary[train_col[i]]

        # Update the columns for test
        for i in np.arange(0, test_col.size):
            if(dictionary.has_key(test_col[i])):
                new_feature_col_test[i] = dictionary[test_col[i]]
            else:
                new_feature_col_test[i] = num_keys_found

        # create a sparse matrix of feature vectors - train
        row = np.arange(0,train_col.size)
        col = new_feature_col_train
        data = np.ones(train_col.size)

        sparse_feature_train = csr_matrix((data, (row,col)), \
                                shape=(train_col.size,len(dictionary) + 1))

        # create a sparse matrix of feature vectors - test
        row = np.arange(0,test_col.size)
        col = new_feature_col_test
        data = np.ones(test_col.size)

        sparse_feature_test = csr_matrix((data, (row,col)), \
                                shape=(test_col.size,len(dictionary) + 1))

        return (sparse_feature_train, sparse_feature_test)

    def train(self):
        self.classifier.fit(self.train_X, self.train_Y)

    def predict(self):
        self.test_Y = self.classifier.predict_proba(self.test_X)[:,1]
        return self.test_Y

    def get_train_accuracy(self):
        train_y_classifier = self.classifier.predict(self.train_X)
        num_samples = train_y_classifier.size
        total_match = sum(train_y_classifier == self.train_Y)
        return total_match/float(num_samples)

    def store_classification_result(self):
        df_out = pd.DataFrame()
        df_out['Id'] = self.test_ID
        df_out['Action'] = self.test_Y
        df_out.to_csv('data/test_result_xgboost.csv',index=False)
