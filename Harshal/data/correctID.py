import pandas as pd

df = pd.read_csv('test_result_xgboost.csv',header=0)
df.Id = df.Id.astype(int)
df.to_csv('test_result_xgboost_with_features.csv',index=False)
