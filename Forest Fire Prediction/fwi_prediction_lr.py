import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Завантаження даних
dataset = pd.read_csv("fire_dataset.csv")

# Видалення першого стовпця
dataset.drop(columns=dataset.columns[0], inplace=True)

# Зберігаємо всі дані для тренування моделі
dataset1 = dataset.drop(["day", "month", "year"], axis=1)

# Розділення на вхідні та вихідні дані
X = dataset1.drop(columns="FWI", axis=1)
y = dataset1["FWI"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Тренування моделі Random Forest
RandomForest_model = RandomForestRegressor()
RandomForest_model.fit(X_train, y_train)
accuracy_RandomForest = RandomForest_model.score(X_test, y_test)
print(f"Точність моделі Random Forest: {accuracy_RandomForest * 100:.2f}%")

# Перевірка, чи є тренувальні дані ненульовими
# print("Унікальні значення FWI в тренувальних даних:", y_train.unique())

# Створення прикладу для передбачення
sample_input = {
    "Temperature": 30,  # Замість цього значення використайте фактичне значення температури
    "Ws": 14,  # Замість цього значення використайте фактичне значення швидкості вітру
    "RH": 54,  # Замість цього значення використайте фактичне значення відносної вологості
    "ISI": 1.3,  # Замість цього значення використайте фактичне значення ISI
    "Rain": 3.1,  # Замість цього значення використайте фактичне значення опадів
}

# Додавання відсутніх стовпців з середніми значеннями (або нулями)
for column in X.columns:
    if column not in sample_input:
        sample_input[column] = X[
            column
        ].mean()  # Використовуємо середнє значення колонки

# Перетворення прикладу на DataFrame з порядком колонок як у X_train
sample_df = pd.DataFrame([sample_input])[X_train.columns]

# Передбачення використовуючи треновану модель Random Forest
predicted_FWI = RandomForest_model.predict(sample_df)
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

# Важливість ознак
feature_importances = RandomForest_model.feature_importances_
features = X.columns

# Візуалізація важливості ознак
# plt.figure(figsize=(10, 6))
# plt.barh(features, feature_importances, color="skyblue")
# plt.xlabel("Важливість ознак")
# plt.ylabel("Ознаки")
# plt.title("Важливість ознак у моделі Decision Tree")
# plt.show()
