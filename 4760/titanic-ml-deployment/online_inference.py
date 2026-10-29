import pandas as pd
import joblib

model = joblib.load("model.pkl")
df = pd.read_csv("train.csv")

# Use one held-out record from the test partition; never pass its Survived label to the model.
from sklearn.model_selection import train_test_split
X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
y = df["Survived"]
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

passenger = X_test.iloc[[0]]
prediction = int(model.predict(passenger)[0])
print("Online inference — one unseen passenger")
print(passenger.to_string(index=False))
print("Prediction:", "Survived (1)" if prediction == 1 else "Did not survive (0)")
