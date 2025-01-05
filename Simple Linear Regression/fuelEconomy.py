import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the data
fuelEconomy_df = pd.read_csv("FuelEconomy.csv")

# Prepare the data
X = fuelEconomy_df["Fuel Economy (MPG)"]
y = fuelEconomy_df["Horse Power"]

X = np.array(X).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

# Standardize the features
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

# Standardize the target
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
simpleLinearRegression_model = LinearRegression()
simpleLinearRegression_model.fit(X_train, y_train)

# Calculate the accuracy
accuracy_LinearRegression = simpleLinearRegression_model.score(X_test, y_test)
print(
    f"Accuracy of Simple Linear Regression Model: {accuracy_LinearRegression * 100:.2f}%"
)

# Use the trained model to generate predictions
Fuel_Economy = np.array([32]).reshape(-1, 1)

# Scale the fuel economy input
Fuel_Economy_scaled = scaler_x.transform(Fuel_Economy)

# Predict horsepower
Horse_Power_scaled = simpleLinearRegression_model.predict(Fuel_Economy_scaled)

# Inverse transform to get the original horsepower scale
Horse_Power = scaler_y.inverse_transform(Horse_Power_scaled.reshape(-1, 1))
print(
    f"Predicted Fuel Economy of {Fuel_Economy[0][0]} MPG with Horse Power: {Horse_Power[0][0]:.1f} HP"
)

# Plot the data
plt.scatter(
    scaler_x.inverse_transform(X_test), scaler_y.inverse_transform(y_test), color="gray"
)
plt.plot(
    scaler_x.inverse_transform(X_test),
    scaler_y.inverse_transform(simpleLinearRegression_model.predict(X_test)),
    color="red",
)
plt.title("Fuel Economy vs Horse Power")
plt.ylabel("Horse Power")
plt.xlabel("Fuel Economy (MPG)")
plt.legend(["Training set", "Test set"])
plt.show()  # Display the plot of the model on the test set

g = sns.PairGrid(fuelEconomy_df)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
plt.show()
