import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

model = joblib.load("model.pkl")
df = pd.read_csv("train.csv")
X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
y = df["Survived"]
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

batch = X_test.iloc[1:21].copy()
results = batch.copy()
results["Predicted_Survived"] = model.predict(batch).astype(int)
results.to_csv("batch_predictions.csv", index=False)
print("Batch inference — predictions for 20 held-out passengers")
print(results.to_string(index=False))
print("Saved results to batch_predictions.csv")
