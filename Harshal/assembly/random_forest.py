from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import *
from feature_creator import Feature_Handler
from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self,K,dirName,criterion='entropy'):
        self.K = K
        self.criterion = criterion
        self.handler = Feature_Handler()
        self.load_data_from_dir(dirName)
        self.rf_model = RandomForestClassifier()
        self.classifier = self.cross_validator()
        self.train()
        self.predict()

    def cross_validator(self):
        parameters = {
            'n_estimators' : [150,300,500,700,1000],
            'criterion' : [self.criterion],
            'max_features': [None,'auto', 'sqrt', 'log2'],
            'class_weight' : ['balanced']
        }

        clf = GridSearchCV(self.rf_model,parameters, n_jobs=4,\
            cv=StratifiedKFold(self.train_Y, n_folds=self.K, shuffle=True), \
            verbose=2, refit=True)

        return clf

    def load_data_from_dir(self,dirName):
        df_train, df_test = self.handler.load_data_from_dir(dirName)
        # Add the relevant features
        #feature_set = df_train.columns.values[3:-1]
        #feature_set = []
        #df_train, df_test = self.handler.feature_adder(df_train,df_test,feature_set)
        # Save the new train and test data
        #self.handler.feature_saver(df_train,df_test)
        # Get train_X, train_y, test_X, test_ID
        self.train_X,self.train_Y,self.test_X,self.test_ID = \
                        self.handler.get_feature_labels(df_train, df_test)


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

    def store_classification_result(self,out_file):
        df_out = pd.DataFrame()
        df_out['Id'] = self.test_ID
        df_out.Id.astype(int)
        df_out['Action'] = self.test_Y
        df_out.to_csv(out_file,index=False)
