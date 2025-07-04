import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
    roc_auc_score
from sklearn.ensemble import RandomForestClassifier

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
# тут масштабування не обовʼязкове для RandomForest, бо дерево не чутливе до масштабів ознак.


# 9. Вибір моделі
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 10. Навчання
rf_model.fit(X_train, y_train)


# 11. Передбачення
y_pred_rf = rf_model.predict(X_test)


# 12. Оцінка
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"RandomForest accuracy: {acc_rf:.2f}")

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("Classification report:")
print(classification_report(y_test, y_pred_rf))

# ROC-AUC
y_proba_rf = rf_model.predict_proba(X_test)[:,1]
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
print(f"ROC-AUC: {roc_auc_rf:.2f}")
