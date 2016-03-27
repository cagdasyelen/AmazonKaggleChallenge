import xgboost as xgb
from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix,hstack
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import *

#Stacking using Shobhit's LR results and putting them into xgboost model

class XgBase:
    def __init__(self):
        self.load_data()
        self.xgb_model = xgb.XGBClassifier()
        self.classifier = self.cross_validator()
        self.train()
        self.predict()

    def load_data(self):
        train_csv = './data/train.csv'
        test_csv = './data/test.csv'
        t1_csv = './data/stacking/lr_balanced_onehot_training.csv'
        l1_csv = './data/stacking/lr_balanced_onehot.csv'
        df_train = pd.read_csv(train_csv, header=0)
        df_test = pd.read_csv(test_csv, header=0)
        df_t1 = pd.read_csv(t1_csv, header=0)
        df_l1 = pd.read_csv(l1_csv, header=0)
        arr_train = df_train.values
        arr_test = df_test.values
        arr_t1 = df_t1.values
        arr_l1 = df_l1.values
        self.train_X = arr_train[0::,1::]
        self.train_Y = arr_train[0::, 0]
        self.test_X = arr_test[0::, 1::]
        self.test_ID = arr_test[0::,0]
        self.t1 = arr_t1[0::,1]
        self.l1 = arr_l1[0::,1]
        self.train_X_added = np.zeros((len(self.train_X), len(self.train_X[0])+1))
        self.test_X_added = np.zeros((len(self.test_X), len(self.test_X[0])+1))
        for i in range(len(self.train_X)):
            self.train_X_added[i][9] = self.t1[i]
            self.test_X_added[i][9] = self.l1[i]


    def cross_validator(self):
        parameters = {
            'objective':['binary:logistic'],
            'max_depth' :[6,8,10,12],
            'n_estimators' : [300,500,800,1000,1400],
            'colsample_bytree' : [0.5,0.7,1]
        }
        clf = GridSearchCV(self.xgb_model,parameters, n_jobs=2,\
            cv=StratifiedKFold(self.train_Y, n_folds=5, shuffle=True), \
            verbose=2, refit=True)

        return clf

    def train(self):
        self.classifier.fit(self.train_X_added, self.train_Y)

    def predict(self):
        self.test_Y = self.classifier.predict_proba(self.test_X)

    def get_training_accuracy(self):
        return (self.classifier.score(self.train_X_added, self.train_Y))

    def store_result(self):
        df_out = pd.DataFrame()
        self.test_ID=self.test_ID.astype(int)
        df_out['Id'] = self.test_ID
        df_out['Action'] = self.test_Y[0::,1]
        df_out.to_csv('./data/results/stacking_result1hot.csv',index=False)

