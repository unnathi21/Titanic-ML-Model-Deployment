import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

model = joblib.load("model.pkl")
df = pd.read_csv("train.csv")
X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
y = df["Survived"]
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Streaming inference — 10 held-out passengers, one record at a time")
stream_records = X_test.iloc[21:31]
for index in range(len(stream_records)):
    record = stream_records.iloc[[index]]
    prediction = int(model.predict(record)[0])
    print(f"Incoming record {index + 1}: prediction = {prediction} "
          f"({'Survived' if prediction == 1 else 'Did not survive'})")
    if index < len(stream_records) - 1:
        time.sleep(1)
