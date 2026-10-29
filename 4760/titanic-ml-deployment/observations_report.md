# Observations: ML Model Deployment Strategies

## 1. How does Online Inference differ from Batch Inference?
Online inference predicts one input record when a request arrives and returns a result immediately. Batch inference processes a group of records together, which is useful for large offline jobs and can be more efficient per record.

## 2. What is the difference between Streaming Inference and Batch Inference?
Streaming inference processes each record as it arrives, usually with low latency and continuous processing. Batch inference waits until a group of records is available and processes that group together.

## 3. Why is FastAPI used for model serving?
FastAPI makes it straightforward to expose a trained model through HTTP REST endpoints. It supports request validation with Pydantic, automatic interactive API documentation through Swagger UI, and asynchronous-friendly server handling.

## 4. Why should the same trained model be reused for all inference strategies?
Using the same saved pipeline keeps preprocessing and model parameters consistent, makes results comparable, and avoids training differences between strategies. It also reflects real deployment, where training and inference are separate steps.

## Result
The pipeline is trained once, evaluated on a held-out split, saved using Joblib, and loaded for all inference demonstrations and API predictions.
