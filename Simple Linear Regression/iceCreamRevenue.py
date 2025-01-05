import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the data
iceCreamData_series = pd.read_csv("IceCreamData.csv")

# Display the correlation matrix
correlation_matrix = iceCreamData_series.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Prepare the data
df_revenue = iceCreamData_series.drop(columns=["Temperature"])
df_temp_target = iceCreamData_series["Temperature"]

df_revenue = np.array(df_revenue)
df_temp_target = np.array(df_temp_target).reshape(-1, 1)

print(f"Average temperature: {df_temp_target.mean():.1f}C")
print(f"Max temperature: {df_temp_target.max()}C\n")

# Standardize the features
scaler_x = StandardScaler()
X = scaler_x.fit_transform(df_revenue)

# Standardize the target
scaler_y = StandardScaler()
Y = scaler_y.fit_transform(df_temp_target)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Train the model
simpleLinearRegression_model = LinearRegression(fit_intercept=True)
simpleLinearRegression_model.fit(X_train, y_train)

print(
    f"Coef (m): {simpleLinearRegression_model.coef_}",
    f"\nIntercept (b): {simpleLinearRegression_model.intercept_}\n",
)

# Calculate the accuracy
accuracy_LinearRegression = simpleLinearRegression_model.score(X_test, y_test)
print(
    f"Accuracy of Simple Linear Regression Model: {accuracy_LinearRegression * 100:.2f}%"
)

# Use the trained model to generate predictions
Temp = np.array([23]).reshape(-1, 1)

# Scale the temperature input
Temp_scaled = scaler_y.transform(Temp)

# Predict revenue
Revenue_scaled = simpleLinearRegression_model.predict(Temp_scaled)

# Inverse transform to get the original revenue scale
Revenue = scaler_x.inverse_transform(Revenue_scaled.reshape(-1, 1))

print(
    f"Revenue Predictions = ${Revenue[0][0]:.2f} while the temperature is {Temp[0][0]}C"
)

# Plot the results
plt.scatter(
    scaler_x.inverse_transform(X_test), scaler_y.inverse_transform(y_test), color="gray"
)
plt.plot(
    scaler_x.inverse_transform(X_test),
    scaler_y.inverse_transform(simpleLinearRegression_model.predict(X_test)),
    color="red",
)
plt.title("Revenue vs Temperature (Training set)")
plt.ylabel("Revenue")
plt.xlabel("Temperature")
plt.legend(["Training set", "Test set"])
plt.show()  # Display the plot of the model on the test set
