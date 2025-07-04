import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor

# 2. Завантаження даних
data = {
    "weight_kg": [4.5, 6.0, 5.0, 10.0, 3.5, 12.0, 4.0, 9.0, 4.2, 11.0, 3.8, 8.5, 5.2, 10.5, 4.3, 12.5, 3.9, 9.5, 4.1, 11.5],
    "tail_length_cm": [25, 30, 28, 45, 20, 50, 23, 40, 22, 47, 21, 38, 26, 44, 24, 52, 22, 42, 23, 48],
    "animal": [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
}

df = pd.DataFrame(data)


# 3. Перевірка даних
# print(df.head())
# print(df.info)
# print(df.describe())
# print(df.isnull().sum)


# 4. EDA
# загальні статистики
# print(df.describe())

# розподіл класів
# print(df["animal"].value_counts()) # покаже, скільки собак і скільки котів

# парні графіки
# sns.pairplot(df, hue="animal")
# plt.show()


# 5. Препроцесинг
# перевіримо пропуски: емає пропусків → нічого робити не треба
# print(df.isnull().sum())

# заповнення пропусків якщо є
# df.fillna(df.median(), inplace=True)


# 6. Вибір features & target
X = df[["weight_kg", "tail_length_cm"]]
y = df["animal"]


# 7. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 8. масштабування
scaler = StandardScaler()
# fit() — скейлер «вивчає» дані: обчислює середнє та стандартне відхилення (для StandardScaler)
# transform() — вже використовує ці обчислені параметри, щоб масштабувати
X_train = scaler.fit_transform(X_train)     # навчаємо + масштабуємо
X_test = scaler.transform(X_test)           # тільки масштабуємо


# 9. Вибір моделі
model = LogisticRegression()


# 10. Навчання
model.fit(X_train, y_train)


# 11. Передбачення
y_pred = model.predict(X_test)


# 12. Оцінка
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion matrix:")
print(cm)











