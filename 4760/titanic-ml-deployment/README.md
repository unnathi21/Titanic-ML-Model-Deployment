# Titanic ML Deployment Strategies and FastAPI Model Serving

## Project overview
This assignment trains a classification pipeline using Kaggle's Titanic `train.csv`, evaluates it on a held-out test partition, saves the complete preprocessing/model pipeline with Joblib, and reuses the same saved model for online, batch, streaming, and REST API inference.

## Dataset
Official dataset page: https://www.kaggle.com/competitions/titanic/data  
The supplied `train.csv` is the labelled Titanic competition training dataset (891 rows). The model target is `Survived`.

## Model and evaluation
- Model: Logistic Regression
- Split: stratified 80% training / 20% testing
- Training rows: 712
- Testing rows: 179
- Test accuracy from this run: **80.45%**
- Preprocessing: numeric median imputation and scaling; categorical most-frequent imputation and one-hot encoding
- Saved pipeline: `model.pkl`

`PassengerId`, `Name`, `Ticket`, and `Cabin` are excluded from features. The pipeline includes preprocessing and the classifier.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run all inference tasks
Keep `train.csv` in the project root.

```bash
python train_model.py
python online_inference.py
python batch_inference.py
python streaming_inference.py
```

- Online inference uses one held-out passenger.
- Batch inference predicts 20 different held-out passengers and writes `batch_predictions.csv`.
- Streaming inference processes 10 other held-out passengers one at a time, with a one-second delay between records.
- The scripts share the same deterministic train/test split, and do not use the `Survived` labels as model inputs during inference.

## FastAPI serving
Start the server:

```bash
uvicorn app:app --reload
```

Open Swagger UI: http://127.0.0.1:8000/docs

Test:
- `GET /health`
- `POST /predict` with a valid passenger JSON body
- `POST /predict` with an invalid `Pclass` such as 5; Pydantic should return HTTP 422

In another terminal, run the Python client:

```bash
python client.py
```

Automated API checks (health, valid prediction, invalid request) can be run with:

```bash
python api_test.py
```

## API example request
```json
{
  "Pclass": 3,
  "Sex": "female",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

## Results and evidence
Execution output captures are in `screenshots/`. They are generated from the actual program outputs recorded when this project was run in the build environment. For submission, also take screenshots of Swagger UI `/docs` showing `/health`, a successful `/predict`, and a rejected invalid request on your own computer if your instructor specifically requires browser screenshots.

## Observations
See `observations_report.pdf` for answers to the four assignment questions.\n