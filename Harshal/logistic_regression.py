from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

class LogReg:
    def __init__(self,dirName):
        self.load_data_from_dir(dirName)
        self.classifier = LogisticRegression()
        self.train()
        self.predict()

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
        self.test_ID = df_test.values[0::,0]

    def train(self):
        self.classifier.fit(self.train_X, self.train_Y)

    def predict(self):
        self.test_Y = self.classifier.predict(self.test_X)

    def get_train_accuracy(self):
        return (self.classifier.score(self.train_X, self.train_Y))

    def store_classification_result(self):
        df_out = pd.DataFrame()
        df_out['Id'] = self.test_ID
        df_out['Action'] = self.test_Y
        df_out.to_csv('data/test_result.csv',index=False)
