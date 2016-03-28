import numpy as np
import pandas as pd
import glob

class Ensembler:
    def __init__(self,label_path, ensemble_save_path):
        self.label_path = label_path
        self.ensemble_save_path = ensemble_save_path
        '''
        @NOTE : Add your names csv file names here to extend the Ensembler
        '''
        self.model_csv_list = ['lr_unbalanced_sparse.csv' ,\
                               'rf_entropy.csv',\
                               'rf_gini.csv',\
                               'extra_forest_entropy.csv',\
                               'extra_forest_gini.csv',\
                               'xgboost.csv']

        '''
        @ NOTE : Add your Kaggle Score for that model for the new model to
        extend the assembler
        '''
        self.private_test_acc_list = ['a', 'b', 'c', 'd', 'e', 'f']

        # Loads all csv_paths into the dictionary dataframe_map with key as name
        # of the file as in model_csv_list
        self.dataframe_map = self.load_model_test_out_csv()

        # Runs different ensemblers and save the result
        self.run_different_ensemblers()



    def run_different_ensemblers(self):
        # Ensembler 1: Simply averaging
        self.average_all_auc()

        # Ensembler 2: Weighted averaging based on AUC
        self.average_weighted_auc()

        # Ensembler 3: Simple averaging among the least correlated M model
        # labels : begins at 2
        for i in np.arange(2,len(model_csv_list)):
            self.least_correlated_simple_auc(i)

        # Ensembler 4: Weighted averaging among the least correlated M models :
        # INCLUDES RANKED WEIGHING
        for i in np.arange(2,len(model_csv_list)):
            self.least_correlated_weighted_auc(i)

    def load_model_test_out_csv(self):
        dataframe_map = {}

        for csv_name in self.model_csv_list:
            csv_path = self.label_path + '/' + csv_name
            dataframe_map[csv_name] = pd.read_csv(csv_path, header=0)

        return dataframe_map

    def average_all_auc(self):
        pass

    def average_weighted_auc(self):
        pass

    def least_correlated_simple_auc(self):
        pass

    def least_correlated_weighted_auc(self):
        pass

    def save_ensembled_labels(self,df, model_name):
        save_path = self.ensemble_save_path + '' + model_name
        df.to_csv(save_path, index=False)
