import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier


data = pd.read_csv("creditcard.csv")

columns = ["Amount"]

for i in range(1, 29):
    columns.append("V" + str(i))

X = data[columns]
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, probabilities)

print("ROC-AUC:", round(auc, 3))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

importance = pd.Series(
    model.feature_importances_,
    index=columns
)

importance = importance.sort_values(ascending=False)

print("\nTop 5 Important Features:")
print(importance.head(5))