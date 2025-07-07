# 1. Імпорти
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
    roc_auc_score

# 2. Завантаження даних
df = pd.read_csv("src/IntrovertExtrovertProblem/data/train.csv")

# 3. Перевірка даних
# print(df.head)
# print(df.info)
# print(df.describe())
# print(df.isnull().sum())

# 4. EDA
# розподіл класів
# print(df["Personality"].value_counts())

# sns.pairplot(df, hue="Personality")
# sns.countplot(x="Personality", hue="Going_outside", data=df)
# sns.countplot(x="Personality", hue="Social_event_attendance", data=df)
# sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
# plt.show()

# 5. Препроцесинг
# перевіримо пропуски: емає пропусків → нічого робити не треба
# print(df.isnull().sum())

df["Stage_fear"] = df["Stage_fear"].map({"Yes": 1, "No": 0})
df["Drained_after_socializing"] = df["Drained_after_socializing"].map({"Yes": 1, "No": 0})
df["Personality"] = df["Personality"].map({"Extrovert": 1, "Introvert": 0})

df.fillna(df.median(), inplace=True)

# 6. Вибір features & target
y = df["Personality"]
X = df[[
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency"
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



#======================Submission============================

df_test = pd.read_csv("src/IntrovertExtrovertProblem/data/test.csv")

df_test["Stage_fear"] = df_test["Stage_fear"].map({"Yes": 1, "No": 0})
df_test["Drained_after_socializing"] = df_test["Drained_after_socializing"].map({"Yes": 1, "No": 0})
df_test.fillna(df_test.median(), inplace=True)

X_submission = df_test[[
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency"
]]

X_submission_scaled = scaler.transform(X_submission)

submission_preds = model.predict(X_submission_scaled)

submission = pd.DataFrame({
    "id": df_test["id"],
    "Personality": submission_preds
})
submission["Personality"] = submission["Personality"].map({1: "Extrovert", 0: "Introvert"})
submission.to_csv("src/IntrovertExtrovertProblem/submission.csv", index=False)
