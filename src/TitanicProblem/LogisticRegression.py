import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
    roc_auc_score

# 2. Завантаження даних
df = pd.read_csv("src/TitanicProblem/data/train.csv")


# 3. Перевірка даних
# print(df.head)
# print(df.info)
# print(df.describe())
# print(df.isnull().sum())


# 4. EDA
# розподіл класів
# print(df["Survived"].value_counts())

# sns.pairplot(df, hue="Survived")
# sns.countplot(x="Survived", hue="Sex", data=df)
# sns.countplot(x="Survived", hue="Pclass", data=df)
# sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
# plt.show()


# 5. Препроцесинг
# перевіримо пропуски: емає пропусків → нічого робити не треба
# print(df.isnull().sum())

# Age — має пропуски
# Cabin — майже весь стовпець пустий (можна просто видалити)
# Embarked — кілька пропусків

# - заповнення пропусків
df.fillna({"Age": df["Age"].median()}, inplace=True)      # заповнюю медіаною, бо розподіл віку трохи зміщений (не симетричний)

# print(df["Embarked"].value_counts())
# всього кілька пропусків — можна заповнити найпопулярнішим портом
# найчастіше це 'S'
df.fillna({"Embarked": "S"}, inplace=True)

df.drop("Cabin", axis=1, inplace=True)      # там майже все NaN, і Cabin дуже деталізований, тому його просто викидають

# print(df.isnull().sum())

# - encoding для категоріальних
# Sex → male / female
# Embarked → S / C / Q
# Pclass → це числове, але по суті категорія (1, 2, 3)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)      # тут варто one-hot
# df = pd.get_dummies(df, columns=["Pclass"], drop_first=True)

# print(df.head())


# 6. Вибір features & target
y = df["Survived"]

# ці ознаки прибираємо: PassengerId, Name, Ticket
X = df[[
"Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_Q",
    "Embarked_S"
]]


# 7. Розбиття на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 8. Масштабування
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 9. Вибір моделі
model = LogisticRegression(max_iter=1000, random_state=42)


# 10. Навчання
model.fit(X_train_scaled, y_train)


# 11. Передбачення
y_pred = model.predict(X_test_scaled)

# також можемо отримати ймовірності:(якщо треба, наприклад, для ROC-кривої)
y_proba = model.predict_proba(X_test_scaled)[:, 1]


# 12. Оцінка
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Classification Report includes precision, recall, F1-score, support
report = classification_report(y_test, y_pred)
print("Classification report:")
print(report)

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.2f}")

