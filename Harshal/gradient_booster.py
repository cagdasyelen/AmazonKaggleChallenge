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
        self.xgb_model = xgb.XGBClassifier()
        self.classifier = self.cross_validator()
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
        # Add the relevant features
        df_train = self.add_relevant_features(df_train, True)
        df_test = self.add_relevant_features(df_test, False)
        # Save the new train and test data
        df_train.to_csv('data/train_with_feat.csv', index=False)
        df_test.to_csv('data/test_with_feat.csv', index=False)
        # Get train_X, train_y, test_X
        train = df_train.values
        self.train_X = train[0::,1::]
        self.train_Y = train[0::,0]
        self.test_X = df_test.values[0::,1::]
        self.test_ID = df_test.values[0::,0]


    def add_relevant_features(self,df, isTrain):
        # Add a unity column for counting purposes
        unity_col = np.ones([df.shape[0],1]).astype(int)
        df['UNITY'] = unity_col
        # Get all the features in the original set
        feature_set = df.columns.values[3:-1]
        # Add the [frequency of resources handled by manager of the resources]
        if isTrain:
            self.table_res = df.pivot_table(index=['MGR_ID'], \
                                    columns=['RESOURCE'], values=['UNITY'], \
                                        aggfunc = np.sum, fill_value = 0).sum(1)
            new_col = table[df['MGR_ID']]
        else:
            new_col = []
            for mgr in df['MGR_ID']:
                if mgr in self.table_res.index:
                    new_col.append(self.table_res[mgr])
                else:
                    new_col.append(0)

        df['MGR_RES'] = np.expand_dims(new_col,axis=1)

        # Add the [number of accepted resources by the manager]
        if isTrain:
            self.table_accepted_res = df.pivot_table(index=['MGR_ID'], \
                                    columns=['UNITY'], values=['ACTION'], \
                                        aggfunc = np.sum, fill_value = 0).sum(1)
            ''' @Note that (column,value) = (Unity,Action) === (Resource,Action)'''
            new_col = self.table_rec[df['MGR_ID']]
        else:
            new_col = []
            for mgr in df['MGR_ID']:
                if mgr in self.table_accepted_res.index:
                    new_col.append(self.table_accepted_res[mgr])
                else:
                    new_col.append(0)


        df['MGR_ACCEPTED_RES'] = np.expand_dims(new_col,axis=1)

        # Fraction of roles accepted for each of :
        # 1. ROLE ROLLUP 1
        # 2. ROLE ROLLUP 2
        # 3. ROLE DEPT NAME
        # 4. ROLE TITLE
        # 5. ROLE_FAMILY_DESC
        # 6. ROLE_FAMILY
        # 7. ROLE_CODE

        # dictionary of tables for all features to update the test data for
        # acceptence related features
        if isTrain:
            self.table_feat = {}
            self.table_feat_accepted = {}
        # Loop over all the features
        for feature in feature_set:
            if isTrain:
                # Total handled
                self.table_feat[feature] = df.pivot_table(index=['MGR_ID',feature], \
                                        columns=['RESOURCE'], values=['UNITY'], \
                                            aggfunc = np.sum, fill_value = 0).sum(1)

                # Total Accepted

                self.table_feat_accepted[feature] = df.pivot_table(index=['MGR_ID',feature], \
                                        columns=['UNITY'], values=['ACTION'], \
                                            aggfunc = np.sum, fill_value = 0).sum(1)

            # total -- Done using for loop : PLEASE IMPROVE
            total_col = []
            for mgr,feat in zip(df['MGR_ID'],df[feature]):
                if isTrain or ((mgr,feat) in self.table_feat[feature].index):
                    total_col.append(self.table_feat[feature][mgr,feat])
                else:
                    total_col.append(1)

            total_col = np.array(total_col).astype(float)

            # accepted -- Done using for loop : PLEASE IMPROVE
            accepted_col = []
            for mgr,feat in zip(df['MGR_ID'],df[feature]):
                if isTrain or ((mgr,feat) in self.table_feat_accepted[feature].index):
                    accepted_col.append(self.table_feat_accepted[feature][mgr,feat])
                else:
                    accepted_col.append(0)

            accepted_col = np.array(accepted_col).astype(float)

            # Get the fraction of acceptance
            fraction_col = np.divide(accepted_col,total_col)

            df['MGR_ACCEPT_FRAC_%s' %(feature)] = np.expand_dims(fraction_col,axis=1)


        # Drop the UNITY col
        df.drop('UNITY',axis=1,inplace=True)

        return df

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
