import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mining_df = pd.read_csv("mining_data.csv")
df_iron = mining_df.drop(columns="% Silica Concentrate")
df_iron_target = mining_df["% Silica Concentrate"]

df_iron = np.array(df_iron)
df_iron_target = np.array(df_iron_target)
# reshaping the array
df_iron_target = df_iron_target.reshape(-1, 1)
df_iron_target.shape

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler_x = StandardScaler()
X = scaler_x.fit_transform(df_iron)

scaler_y = StandardScaler()
Y = scaler_y.fit_transform(df_iron_target)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

DecisionTree_model = DecisionTreeRegressor()
DecisionTree_model.fit(X_train, y_train)
accuracy_DecisionTree = DecisionTree_model.score(X_test, y_test)
print(f"Accuracy of Decision Tree Model: {accuracy_DecisionTree * 100:.2f}%")

y_predict = DecisionTree_model.predict(X_test)
plt.plot(y_predict, y_test, "o", color="green")
plt.xlabel("Model Predictions")
plt.ylabel("True Values")
plt.title("Decision Tree Model Predictions")
plt.show()

# Reshape y_predict to 2D array
y_predict_reshaped = y_predict.reshape(-1, 1)
y_predict_orig = scaler_y.inverse_transform(y_predict_reshaped)

# Reshape y_test to 2D array
y_test_reshaped = y_test.reshape(-1, 1)
y_test_orig = scaler_y.inverse_transform(y_test_reshaped)

plt.plot(y_test_orig, y_predict_orig, "^", color="r")
plt.xlabel("Model Predictions")
plt.ylabel("True Values")
plt.show()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

k = X_test.shape[1]
n = len(X_test)
RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)), ".3f"))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

print(
    "RMSE =",
    RMSE,
    "\nMSE =",
    MSE,
    "\nMAE =",
    MAE,
    "\nR2 =",
    r2,
    "\nAdjusted R2 =",
    adj_r2,
)
