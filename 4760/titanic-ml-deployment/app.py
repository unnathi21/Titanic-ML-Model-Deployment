from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

MODEL_PATH = Path(__file__).with_name("model.pkl")
model = joblib.load(MODEL_PATH)
app = FastAPI(title="Titanic Survival Prediction API", version="1.0")

class Passenger(BaseModel):
    model_config = ConfigDict(extra="forbid")
    Pclass: int = Field(ge=1, le=3, description="Passenger class: 1, 2, or 3")
    Sex: str = Field(pattern="^(male|female)$")
    Age: float | None = Field(default=None, ge=0, le=120)
    SibSp: int = Field(ge=0, le=20)
    Parch: int = Field(ge=0, le=20)
    Fare: float = Field(ge=0)
    Embarked: str | None = Field(default=None, pattern="^(S|C|Q)$")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(passenger: Passenger):
    # The fields and order match the features used when training.
    row = pd.DataFrame([passenger.model_dump()])
    prediction = int(model.predict(row)[0])
    return {
        "prediction": prediction,
        "label": "Survived" if prediction == 1 else "Did not survive"
    }
