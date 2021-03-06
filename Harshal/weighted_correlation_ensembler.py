import numpy as np
import pandas as pd
import glob
from scipy.stats.stats import pearsonr
from operator import itemgetter

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
                               'xgboost.csv',\
                               'test_result_xgboost_base_case.csv',\
                               'result_default_set.csv',\
                               'c2_base_result.csv',\
                               'result_with_diff_param1.csv',\
                               '00-neg_bal_mix.csv',\
                               '01-lr_balanced_onehot_deg3_greedy_9.csv',\
                               '02-lr_balanced_onehot_deg2-3_top15_noCODE.csv',\
                               '03-lr_balanced_onehot_deg2-3_greedy_7_noROLLUP.csv',\
                               '04-lr_balanced_onehot_deg2-3_greedy_13_noCODE.csv',\
                               '05-lr_balanced_onehot_deg2_greedy_12_noCODE.csv']

        '''
        @ NOTE : Add your Kaggle Score for that model for the new model to
        extend the assembler
        '''''' 0.80865, 0.81003, '''
        self.private_test_acc_list = [0.89011, 0.86182, 0.86584, 0.80865, 0.81003,\
                                      0.86753, 0.87476, 0.87266, 0.87202, 0.87021,\
                                      0.89129, 0.89011, 0.88755, 0.88467, 0.88413,\
                                      0.88410]
        self.model_rank = self.rank_scores()

        # Loads all csv_paths into the dictionary dataframe_map with key as name
        # of the file as in model_csv_list
        self.dataframe_map = self.load_model_test_out_csv()


    def rank_scores(self):
        map_to_index = {}

        for i in range(0, len(self.private_test_acc_list)):
            map_to_index[self.private_test_acc_list[i]] = i

        # sort the list
        self.private_test_acc_list.sort()

        # Now get the rank
        model_rank = np.zeros(len(self.private_test_acc_list))
        for i in range(0, len(self.private_test_acc_list)):
            model_rank[map_to_index[self.private_test_acc_list[i]]] = i

        return model_rank



    def run_different_ensemblers(self):
        # Ensembler 1: Simply averaging
        self.average_all_auc()

        # Ensembler 2: Weighted averaging based on AUC
        self.average_weighted_auc()

        # Ensembler 3: Simple averaging among the least correlated M model
        self.least_correlated_simple_auc(3)

        # Ensembler 4: Weighted averaging among the least correlated M models
        self.least_correlated_weighted_auc(3)


    def load_model_test_out_csv(self):
        dataframe_map = {}

        for csv_name in self.model_csv_list:
            csv_path = self.label_path + '/' + csv_name
            dataframe_map[csv_name] = pd.read_csv(csv_path, header=0)

        return dataframe_map

    def average_all_auc(self):
        sum_all_models = np.zeros([58921,1])
        for model in self.model_csv_list:
            df = self.dataframe_map[model]
            df_arr = df.values
            label = df_arr[:,1]
            label = np.expand_dims(label,axis = 1)
            sum_all_models += label

        average_auc_labels = sum_all_models/ len(self.model_csv_list)
        df_average = pd.DataFrame()
        df_average['Id'] = self.dataframe_map[self.model_csv_list[0]]['Id']
        df_average['Action'] = average_auc_labels
        self.save_ensembled_labels(df_average, 'average_all_auc')

    def average_weighted_auc(self):
        sum_all_models = np.zeros([58921,1])
        total_rank_sum = 0

        counter = 0
        for model in self.model_csv_list:
            label = self.dataframe_map[model].values[:,1]
            label = np.expand_dims(label,axis = 1)
            sum_all_models += (label * (self.model_rank[counter]))
            total_rank_sum += self.model_rank[counter]
            counter += 1

        average_weighted_auc_labels = sum_all_models/ total_rank_sum
        df_weighted_average = pd.DataFrame()
        df_weighted_average['Id'] = self.dataframe_map[self.model_csv_list[0]]['Id']
        df_weighted_average['Action'] = average_weighted_auc_labels
        self.save_ensembled_labels(df_weighted_average, 'average_weighted_all_auc')

    def least_correlated_simple_auc(self,num_models):
        least_correlated_list = self.find_least_correlated_set(num_models)
        sum_all_models = np.zeros([58921,1])
        for model in least_correlated_list:
            df = self.dataframe_map[model]
            df_arr = df.values
            label = df_arr[:,1]
            label = np.expand_dims(label,axis = 1)
            sum_all_models += label

        average_auc_labels = sum_all_models/ len(least_correlated_list)
        df_average = pd.DataFrame()
        df_average['Id'] = self.dataframe_map[least_correlated_list[0]]['Id']
        df_average['Action'] = average_auc_labels
        self.save_ensembled_labels(df_average, 'average_all_auc_least_correlated')

    def find_least_correlated_set(self,n_models):
        model_pair_uncorrelation_list = []
        for i in range(0, len(self.model_csv_list) - 1):
            for j in range(i + 1, len(self.model_csv_list)):
                df_model_1 = self.dataframe_map[self.model_csv_list[i]]
                df_model_2 = self.dataframe_map[self.model_csv_list[j]]
                model_1 = df_model_1.values[:,1]
                model_2 = df_model_2.values[:,1]
                uncorrelation_coeff = pearsonr(model_1,model_2)[1]
                model_pair_uncorrelation_list.append((self.model_csv_list[i],\
                                                      self.model_csv_list[j],\
                                                      uncorrelation_coeff))

        # sort the uncorrelation in decreasing order
        model_pair_uncorrelation_list = sorted(model_pair_uncorrelation_list,\
                                               key = itemgetter(2),\
                                               reverse=True)

        # Get best n_models
        best_models = set()
        # Add the best model so far
        for model_1,model_2, uncorr in model_pair_uncorrelation_list:
            if len(best_models) <= n_models - 2:
                best_models.add(model_1)
                best_models.add(model_2)
            elif len(best_models) == n_models - 1:
                if model_1 in best_models:
                    best_models.add(model_2)
                elif model_2 in best_models:
                    best_models.add(model_1)
                else:
                    best_models.add(model_1)
            else:
                break

        return list(best_models)


    def least_correlated_weighted_auc(self,n_models):
        least_correlated_list = self.find_least_correlated_set(n_models)
        sum_all_models = np.zeros([58921,1])
        total_rank_sum = 0

        for model in least_correlated_list:
            label = self.dataframe_map[model].values[:,1]
            label = np.expand_dims(label,axis = 1)
            sum_all_models += (label * (self.model_rank[self.model_csv_list.index(model)]))
            total_rank_sum += self.model_rank[self.model_csv_list.index(model)]

        average_weighted_auc_labels = sum_all_models/ total_rank_sum
        df_weighted_average = pd.DataFrame()
        df_weighted_average['Id'] = self.dataframe_map[least_correlated_list[0]]['Id']
        df_weighted_average['Action'] = average_weighted_auc_labels
        self.save_ensembled_labels(df_weighted_average, 'average_weighted_all_auc_least_correlated')

    def save_ensembled_labels(self,df, ensemble_name):
        save_path = self.ensemble_save_path + '/' + ensemble_name + '.csv'
        df.to_csv(save_path, index=False)


def main():
    ensembler = Ensembler('data/ensemble_results','data/ensembled_labels')
    ensembler.run_different_ensemblers()

if __name__ == "__main__":
    main()
