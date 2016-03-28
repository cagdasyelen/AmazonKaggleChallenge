import pandas as pd
import numpy as np


'''

Below listed the various combinations and their scores


#majority7_1.csv	0.90047	0.89423

l1_csv = './data/lr_balanced_onehot_deg2-3_greedy_13.csv'
l2_csv = './data/lr_balanced_onehot_deg2_greedy_13.csv'
l3_csv = './data/lr_balanced_onehot_deg3_greedy_9.csv'
l4_csv = './data/test_lr_balanced_onehot_5feat.csv'
l5_csv = './data/test_result_xgboost_base_case.csv'
l6_csv = './data/result_with_diff_param1.csv'
l7_csv = './data/extra_forest_entropy.csv'

#majority7_2.csv	0.89822	0.89274

l1_csv = './data/01-lr_balanced_onehot_deg3_greedy_9.csv'
l2_csv = './data/02-lr_balanced_onehot_deg2-3_greedy_7_noROLLUP.csv'
l3_csv = './data/05-lr_balanced_onehot_deg2_greedy_12_noCODE.csv'
l4_csv = './data/04-lr_balanced_onehot_deg2-3_greedy_13_noCODE.csv'
l5_csv = './data/test_result_xgboost_base_case.csv'
l6_csv = './data/result_with_diff_param1.csv'
l7_csv = './data/extra_forest_entropy.csv'

#majority8.csv	0.90104	0.89450

l1_csv = './data/lr_balanced_onehot_deg2-3_greedy_13.csv'
l2_csv = './data/lr_balanced_onehot_deg2_greedy_13.csv'
l3_csv = './data/lr_balanced_onehot_deg3_greedy_9.csv'
l4_csv = './data/test_lr_balanced_onehot_5feat.csv'
l5_csv = './data/test_result_xgboost_base_case.csv'
l6_csv = './data/result_with_diff_param1.csv'
l7_csv = './data/rf_entropy.csv'
l8_csv = './data/rf_gini.csv'


#majority10.csv	0.90091	0.89439

l1_csv = './data/01-lr_balanced_onehot_deg3_greedy_9.csv'
l2_csv = './data/02-lr_balanced_onehot_deg2-3_greedy_7_noROLLUP.csv'
l3_csv = './data/03-lr_balanced_onehot_deg2-3_top15_noCODE.csv'
l4_csv = './data/04-lr_balanced_onehot_deg2-3_greedy_13_noCODE.csv'
l5_csv = './data/05-lr_balanced_onehot_deg2_greedy_12_noCODE.csv'
l6_csv = './data/06-lr_balanced_onehot_5feat.csv'
l7_csv = './data/test_result_xgboost_base_case.csv'
l8_csv = './data/result_with_diff_param1.csv'
l9_csv = './data/rf_entropy.csv'
l10_csv = './data/rf_gini.csv'

#majority9_1.csv	0.90120	0.89467

l1_csv = './data/01-lr_balanced_onehot_deg3_greedy_9.csv'
l2_csv = './data/02-lr_balanced_onehot_deg2-3_greedy_7_noROLLUP.csv'
l4_csv = './data/04-lr_balanced_onehot_deg2-3_greedy_13_noCODE.csv'
l5_csv = './data/05-lr_balanced_onehot_deg2_greedy_12_noCODE.csv'
l6_csv = './data/06-lr_balanced_onehot_5feat.csv'
l7_csv = './data/test_result_xgboost_base_case.csv'
l8_csv = './data/result_with_diff_param1.csv'
l9_csv = './data/rf_entropy.csv'
l10_csv = './data/rf_gini.csv'

'''

l1_csv = './data/01-lr_balanced_onehot_deg3_greedy_9.csv'
l2_csv = './data/02-lr_balanced_onehot_deg2-3_greedy_7_noROLLUP.csv'
l3_csv = './data/03-lr_balanced_onehot_deg2-3_top15_noCODE.csv'
l4_csv = './data/04-lr_balanced_onehot_deg2-3_greedy_13_noCODE.csv'
l5_csv = './data/05-lr_balanced_onehot_deg2_greedy_12_noCODE.csv'
l6_csv = './data/06-lr_balanced_onehot_5feat.csv'
l7_csv = './data/test_result_xgboost_base_case.csv'
l8_csv = './data/result_with_diff_param1.csv'
l9_csv = './data/rf_entropy.csv'
l10_csv = './data/rf_gini.csv'

df_l1 = pd.read_csv(l1_csv, header=0)
df_l2 = pd.read_csv(l2_csv, header=0)
df_l3 = pd.read_csv(l3_csv, header=0)
df_l4 = pd.read_csv(l4_csv, header=0)
df_l5 = pd.read_csv(l5_csv, header=0)
df_l6 = pd.read_csv(l6_csv, header=0)
df_l7 = pd.read_csv(l7_csv, header=0)
df_l8 = pd.read_csv(l8_csv, header=0)
df_l9 = pd.read_csv(l9_csv, header=0)
df_l10 = pd.read_csv(l10_csv, header=0)


arr_l1 = df_l1.values
arr_l2 = df_l2.values
arr_l3 = df_l3.values
arr_l4 = df_l4.values
arr_l5 = df_l5.values
arr_l6 = df_l6.values
arr_l7 = df_l7.values
arr_l8 = df_l8.values
arr_l9 = df_l9.values
arr_l10 = df_l10.values

l1= arr_l1[:,1]
l2=arr_l2[:,1]
l3= arr_l3[:,1]
l4=arr_l4[:,1]
l5=arr_l5[:,1]
l6=arr_l6[:,1]
l7=arr_l7[:,1]
l8=arr_l8[:,1]
l9=arr_l9[:,1]
l10=arr_l10[:,1]

test_ID = arr_l2[:,0]
l = np.zeros(len(l2))




for i in range(len(l)):
    l[i] = (l1[i] + l2[i]  + l4[i] + l5[i] +l6[i] +l7[i] + l8[i] +l9[i] + l10[i])/9


test_ID = test_ID.astype(int)
df_out = pd.DataFrame()
df_out['Id'] = test_ID
df_out['Action'] = l
df_out.to_csv('./results/majority9_1.csv',index=False, dtype=int)
print 'done'
