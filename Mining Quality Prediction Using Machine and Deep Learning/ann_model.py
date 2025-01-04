import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

mining_df = pd.read_csv("mining_data.csv")
df_iron = mining_df.drop(columns="% Silica Concentrate")
df_iron_target = mining_df["% Silica Concentrate"]
print(df_iron.shape)
print(df_iron_target.shape)

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

optimizer = Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
)
ANN_model = Sequential()
ANN_model.add(Dense(250, input_dim=22, kernel_initializer="normal", activation="relu"))
ANN_model.add(Dense(500, activation="relu"))
ANN_model.add(Dropout(0.1))
ANN_model.add(Dense(1000, activation="relu"))
ANN_model.add(Dropout(0.1))
ANN_model.add(Dense(1000, activation="relu"))
ANN_model.add(Dropout(0.1))
ANN_model.add(Dense(500, activation="relu"))
ANN_model.add(Dropout(0.1))
ANN_model.add(Dense(250, activation="relu"))
ANN_model.add(Dropout(0.1))
ANN_model.add(Dense(250, activation="relu"))
ANN_model.add(Dropout(0.1))
ANN_model.add(Dense(1, activation="linear"))
ANN_model.compile(loss="mse", optimizer="adam")
ANN_model.summary()

history = ANN_model.fit(X_train, y_train, validation_split=0.2, epochs=2)

result = ANN_model.evaluate(X_test, y_test)
accuracy_ANN = 1 - result
print(f"Accuracy : {accuracy_ANN}")

history.history.keys()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train_loss", "val_loss"], loc="upper right")
plt.show()
