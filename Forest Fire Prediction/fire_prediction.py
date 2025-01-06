import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

dataset = pd.read_csv("fire_dataset.csv")
# Drop the first column
dataset.drop(columns=dataset.columns[0], inplace=True)

dataset1 = dataset.drop(["day", "month", "year"], axis=1)

temp = dataset["Temperature"]
day = dataset["day"]
month = dataset["month"]
year = dataset["year"]

# Combine day, month, and year into a single date column
dataset["Date"] = pd.to_datetime(dataset[["year", "month", "day"]]).dt.date

# Percentage of fire and no fire
percentage = dataset.Classes.value_counts(normalize=True) * 100
# Plot pie chart of percentage of fire and no fire
# plt.pie(percentage, labels=["Fire", "No fire"], autopct="%1.1f%%")
# plt.title("Percentage of Fire and No Fire")
# plt.show()

# dftemp = dataset.loc[dataset["Region"] == 1]
# sns.countplot(x="month", hue="Classes", data=dataset, ec="black")
# plt.xticks(
#     np.arange(4),
#     [
#         "June",
#         "July",
#         "August",
#         "September",
#     ],
# )
# plt.title("Fire Analysis Month wise for Bejaia Region")
# plt.legend(["No fire", "Fire"])
# plt.show()


# def barchart(feature, xlabel):
#     """
#     Plots a bar chart for the given feature against the fire count.

#     Parameters:
#     feature (str): The feature to plot on the x-axis.
#     xlabel (str): The label for the x-axis.
#     """
#     plt.figure(figsize=[14, 8])
#     by_feature = dataset1.groupby([feature], as_index=False)["Classes"].sum()
#     ax = sns.barplot(
#         legend=False,
#         x=feature,
#         y="Classes",
#         data=by_feature[[feature, "Classes"]],
#         hue="Classes",
#         estimator=sum,
#     )
#     ax.set(xlabel=xlabel, ylabel="Fire Count")
#     plt.show()


# barchart("Temperature", "Temperature")
# barchart("Ws", "Wind Speed in km/hr")
# barchart("RH", "Relative Humidity in %")

# for date, temperature, fire in zip(
#     dataset["Date"], dataset["Temperature"], dataset["Classes"]
# ):
#     print(f"Date: {date}, Temperature: {temperature}, Fire: {fire}")


# Plot histogram of temperature according to date
# plt.figure(figsize=(14, 7))
# sns.histplot(data=dataset, x="Date", y="Temperature", bins=50, kde=True)
# plt.title("Histogram of Temperature According to Date")
# plt.xlabel("Date")
# plt.ylabel("Temperature")
# sns.histplot(
#     data=dataset[dataset["Classes"] == 1],
#     x="Date",
#     y="Temperature",
#     bins=50,
#     kde=True,
#     color="red",
#     label="Fire",
# )
# plt.title("Histogram of Temperature According to Date")
# plt.xlabel("Date")
# plt.ylabel("Temperature")
# plt.show()

specified_temperature = 42  # Replace with the desired temperature value
result = any(
    (dataset["Classes"] == 1) & (dataset["Temperature"] == specified_temperature)
)
print(result)

X = dataset1.drop(columns="FWI", axis=1)
y = dataset1["FWI"]
# df_fwi = np.array(df_fwi)
# df_fwi_target = np.array(df_fwi_target).reshape(-1, 1)

# print(df_fwi.shape, df_fwi_target.shape)

# scaler_x = StandardScaler()
# X = scaler_x.fit_transform(df_fwi)
# scaler_y = StandardScaler()
# Y = scaler_y.fit_transform(df_fwi_target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print(X_train.shape, X_test.shape)
DecisionTree_model = DecisionTreeRegressor()
DecisionTree_model.fit(X_train, y_train)
accuracy_DecisionTree = DecisionTree_model.score(X_test, y_test)
print(f"Accuracy of Decision Tree Model: {accuracy_DecisionTree * 100:.2f}%")

# # Using Pearson Correlation
# plt.figure(figsize=(12, 10))
# cor = X_train.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
# plt.show()


def predict_fire(temperature, wind_speed, humidity):
    """
    Predicts if fire is possible based on temperature, wind speed, and humidity.

    Parameters:
    temperature (float): The temperature value.
    wind_speed (float): The wind speed value.
    humidity (float): The humidity value.

    Returns:
    bool: True if fire is predicted, False otherwise.
    """
    input_data = pd.DataFrame(columns=feature_names)

    # Set the input features
    input_data.loc[0, "Temperature"] = temperature
    input_data.loc[0, "Ws"] = wind_speed
    input_data.loc[0, "RH"] = humidity

    # Assuming other features are set to their mean values
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = X[col].mean()

    # Ensure the column order matches the training data
    input_data = input_data[feature_names]

    prediction = DecisionTree_model.predict(input_data)
    return prediction[0] > 0  # Assuming a threshold of 0 for fire prediction


# Example usage
temperature = 15
wind_speed = 5
humidity = 10
fire_possible = predict_fire(temperature, wind_speed, humidity)
print(f"Fire possible: {fire_possible}")


# y_predict = DecisionTree_model.predict(X_test)
# plt.plot(y_predict, y_test, "o", color="green")
# plt.xlabel("Model Predictions")
# plt.ylabel("True Values")
# plt.title("Decision Tree Model Predictions")
# plt.show()
