import xgboost as xgb
from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix,hstack
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import *

class Feature_Handler:
    def __init__(self):
        pass

    def load_data_from_dir(self,dirName):
        train_file = './'+ dirName + '/train.csv'
        test_file = './' + dirName + '/test.csv'
        df_train = pd.read_csv(train_file, header=0)
        df_test = pd.read_csv(test_file, header=0)
        return (df_train, df_test)

    def feature_adder(self, df_train, df_test, feature_set):
        df_train = self.add_relevant_features(df_train, feature_set, True)
        df_test = self.add_relevant_features(df_test, feature_set, False)
        return (df_train, df_test)

    def feature_saver(self,df_train, df_test):
        df_train.to_csv('data/train_with_feat.csv', index=False)
        df_test.to_csv('data/test_with_feat.csv', index=False)

    def get_feature_labels(self, df_train, df_test):
        train = df_train.values
        train_X = train[0::,1::]
        train_Y = train[0::,0]
        test_X = df_test.values[0::,1::]
        test_ID = df_test.values[0::,0]
        return (train_X, train_Y, test_X, test_ID)

    def resource_per_manager(self, df, isTrain):
        if isTrain:
            self.table_res = df.pivot_table(index=['MGR_ID'], \
                                    columns=['RESOURCE'], values=['UNITY'], \
                                        aggfunc = np.sum, fill_value = 0).sum(1)
            new_col = self.table_res[df['MGR_ID']]
        else:
            new_col = []
            for mgr in df['MGR_ID']:
                if mgr in self.table_res.index:
                    new_col.append(self.table_res[mgr])
                else:
                    new_col.append(0)

        df['MGR_RES'] = np.expand_dims(new_col,axis=1)
        return df

    def accepted_resource_per_manager(self,df,isTrain):
        if isTrain:
            self.table_accepted_res = df.pivot_table(index=['MGR_ID'], \
                                    columns=['UNITY'], values=['ACTION'], \
                                        aggfunc = np.sum, fill_value = 0).sum(1)
            ''' @Note that (column,value) = (Unity,Action) === (Resource,Action)'''
            new_col = self.table_accepted_res[df['MGR_ID']]
        else:
            new_col = []
            for mgr in df['MGR_ID']:
                if mgr in self.table_accepted_res.index:
                    new_col.append(self.table_accepted_res[mgr])
                else:
                    new_col.append(0)


        df['MGR_ACCEPTED_RES'] = np.expand_dims(new_col,axis=1)

        return df


    def total_and_accepted_table(self, df, isTrain,feature):
        if isTrain:
            # Total handled
            self.table_feat[feature] = df.pivot_table(\
                                    index=['MGR_ID',feature], \
                                    columns=['RESOURCE'], \
                                    values=['UNITY'], \
                                    aggfunc = np.sum, \
                                    fill_value = 0).sum(1)

            # Total Accepted
            self.table_feat_accepted[feature] = df.pivot_table(\
                                    index=['MGR_ID',feature], \
                                    columns=['UNITY'], \
                                    values=['ACTION'], \
                                    aggfunc = np.sum, \
                                    fill_value = 0).sum(1)
                                    

    def add_relevant_features(self, df, feature_set, isTrain):
        # Add a unity column for counting purposes
        unity_col = np.ones([df.shape[0],1]).astype(int)
        df['UNITY'] = unity_col
        # Add the [frequency of resources handled by manager of the resources]
        df = self.resource_per_manager(df,isTrain)
        # Add the [number of accepted resources by the manager]
        df = self.accepted_resource_per_manager(df,isTrain)
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
            # Create a table of total and accepted for (mgr, feature) tuple
            self.total_and_accepted_table(df,isTrain,feature)
            ##### total -- Done using for loop : PLEASE IMPROVE
            total_col = []
            for mgr,feat in zip(df['MGR_ID'],df[feature]):
                if isTrain or ((mgr,feat) in self.table_feat[feature].index):
                    total_col.append(self.table_feat[feature][mgr,feat])
                else:
                    total_col.append(1)
            total_col = np.array(total_col).astype(float)

            ##### accepted -- Done using for loop : PLEASE IMPROVE
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
