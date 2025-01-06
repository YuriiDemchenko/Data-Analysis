import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Loading the dataset
dataset = pd.read_csv("fire_dataset.csv")

# Removing the first column
dataset.drop(columns=dataset.columns[0], inplace=True)

# Dataset after dropping the columns
dataset1 = dataset.drop(["day", "month", "year"], axis=1)

# Splitting the dataset into the training and testing dataset
X = dataset1.drop(columns="FWI", axis=1)
y = dataset1["FWI"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Training the Decision Tree model
DecisionTree_model = DecisionTreeRegressor()
DecisionTree_model.fit(X_train, y_train)
accuracy_DecisionTree = DecisionTree_model.score(X_test, y_test)
print(f"Точність моделі Decision Tree: {accuracy_DecisionTree * 100:.2f}%")

# Checking the unique values of FWI in the training data
print("Унікальні значення FWI в тренувальних даних:", y_train.unique())

# A sample input for prediction
sample_input = {
    "Temperature": 25,  # Temperature in Celsius
    "Ws": 13,  # Wind Speed in km/hr
}

# Adding the mean value of the columns that are not in the sample input
for column in X.columns:
    if column not in sample_input:
        sample_input[column] = X[column].mean()  # Adding the mean value of the column

# Transforming the sample input to match the input features of the model
sample_df = pd.DataFrame([sample_input])[X_train.columns]

# Fire Weather Index (FWI) Index which ranges between 1 to 31.1,
# here 0-3 has lower chance of Forest fires and 3-25 FWI has higher chance of forest fires.
# Predicting the FWI using the trained Decision Tree model
predicted_FWI = DecisionTree_model.predict(sample_df)
if 0 <= predicted_FWI[0] <= 3:
    print(
        f"Прогнозований FWI: {predicted_FWI[0]:.2f} (Низька ймовірність лісових пожеж)"
    )
elif 3 < predicted_FWI[0] <= 25:
    print(
        f"Прогнозований FWI: {predicted_FWI[0]:.2f} (Висока ймовірність лісових пожеж)"
    )
else:
    print(
        f"Прогнозований FWI: {predicted_FWI[0]:.2f} (Дуже висока ймовірність лісових пожеж)"
    )
