from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

health = client.get("/health")
print("GET /health:", health.status_code, health.json())

valid_payload = {
    "Pclass": 3, "Sex": "female", "Age": 22, "SibSp": 1,
    "Parch": 0, "Fare": 7.25, "Embarked": "S"
}
valid = client.post("/predict", json=valid_payload)
print("POST /predict (valid):", valid.status_code, valid.json())

invalid_payload = {**valid_payload, "Pclass": 5}
invalid = client.post("/predict", json=invalid_payload)
print("POST /predict (invalid Pclass=5):", invalid.status_code, invalid.json())

assert health.status_code == 200
assert valid.status_code == 200
assert invalid.status_code == 422
print("All API checks passed.")
