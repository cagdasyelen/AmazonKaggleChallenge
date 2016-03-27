import pandas as pd
import numpy as np




l1_csv = 'cagdas_xgb_base.csv'
l2_csv = 'shobhit_balanced_onehot.csv'

df_l1 = pd.read_csv(l1_csv, header=0)
df_l2 = pd.read_csv(l2_csv, header=0)

arr_l1 = df_l1.values
arr_l2 = df_l2.values
l1= arr_l1[:,1]
l2=arr_l2[:,1]
test_ID = arr_l2[:,0]
l3 = np.zeros(len(l2))
print l2

for i in range(len(l3)):
    l3[i] = (l1[i] + l2[i])/2
test_ID = test_ID.astype(int)
df_out = pd.DataFrame()
df_out['Id'] = test_ID
df_out['Action'] = l3
df_out.to_csv('majority.csv',index=False, dtype=int)

