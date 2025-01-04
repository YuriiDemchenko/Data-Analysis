import pandas as pd
import numpy as np

mining_df = pd.read_csv('mining_data.csv')
df_iron = mining_df.drop(columns = '% Silica Concentrate')
df_iron_target = mining_df['% Silica Concentrate']
print(df_iron.shape)
print(df_iron_target.shape)

df_iron = np.array(df_iron)
df_iron_target = np.array(df_iron_target)
# reshaping the array
df_iron_target = df_iron_target.reshape(-1,1)
df_iron_target.shape

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(df_iron)

scaler_y = StandardScaler()
Y = scaler_y.fit_transform(df_iron_target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)    

print("X TEST: ",X_test.shape)
print("X TRAIN: ",X_train.shape)