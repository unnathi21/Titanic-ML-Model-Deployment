import requests

BASE_URL = "http://127.0.0.1:8000"

health_response = requests.get(f"{BASE_URL}/health", timeout=5)
print("Health status:", health_response.status_code, health_response.json())

passenger = {
    "Pclass": 3,
    "Sex": "female",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}
response = requests.post(f"{BASE_URL}/predict", json=passenger, timeout=5)
print("Prediction status:", response.status_code)
print("Prediction response:", response.json())
response.raise_for_status()
