import pandas as pd
import numpy as np




l1_csv = 'lr_balanced_onehot_deg2-3_greedy_13.csv'
l2_csv = 'lr_balanced_onehot_deg2_greedy_13.csv'
l3_csv = 'lr_balanced_onehot_deg3_greedy_9.csv'
l4_csv = 'test_lr_balanced_onehot_5feat.csv'
l5_csv = 'test_result_xgboost_base_case.csv'

df_l1 = pd.read_csv(l1_csv, header=0)
df_l2 = pd.read_csv(l2_csv, header=0)
df_l3 = pd.read_csv(l4_csv, header=0)
df_l4 = pd.read_csv(l4_csv, header=0)
df_l5 = pd.read_csv(l5_csv, header=0)

arr_l1 = df_l1.values
arr_l2 = df_l2.values
arr_l3 = df_l3.values
arr_l4 = df_l4.values
arr_l5 = df_l5.values

l1= arr_l1[:,1]
l2=arr_l2[:,1]
l3= arr_l3[:,1]
l4=arr_l4[:,1]
l5=arr_l5[:,1]

test_ID = arr_l2[:,0]
l = np.zeros(len(l2))
labels = np.zeros(len(l2))



for i in range(len(l)):
    l[i] = (l1[i] + l2[i] + l3[i] + l4[i] + l5[i])/5
    if(l[i] > 0.75 ):
        labels[i] = 1

test_ID = test_ID.astype(int)
labels = labels.astype(int)
df_out = pd.DataFrame()
df_out['Id'] = test_ID
df_out['Action'] = labels
df_out.to_csv('majority_threshold_0.75',index=False, dtype=int)

