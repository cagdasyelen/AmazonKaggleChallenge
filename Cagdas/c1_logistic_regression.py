import numpy as np
import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression



class LogReg:
    def __init__(self):
        self.load_data()
        self.clf = LogisticRegression(class_weight = 'balanced')
        self.train()
        self.predict()

    def load_data(self):
        train_csv = './data/train.csv'
        test_csv = './data/test.csv'
        df_train = pd.read_csv(train_csv, header=0)
        df_test = pd.read_csv(test_csv, header=0)
        arr_train = df_train.values
        arr_test = df_test.values
        self.train_X = arr_train[0::,1::]
        self.train_Y = arr_train[0::, 0]
        self.test_X = arr_test[0::, 1::]
        self.test_ID = arr_test[0::,0]

    def train(self):
        self.clf.fit(self.train_X, self.train_Y)

    def predict(self):
        self.test_Y = self.clf.predict_proba(self.test_X)

    def get_training_accuracy(self):
        return (self.clf.score(self.train_X, self.train_Y))

    def store_result(self):
        df_out = pd.DataFrame()
        df_out['Id'] = self.test_ID
        df_out['Action'] = self.test_Y[0::,1]
        df_out.to_csv('./data/c1_result.csv',index=False)







