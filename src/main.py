from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI()

# Load the pre-trained model
model = joblib.load('src/model.joblib')


class InputData(BaseModel):
    features: List[List[float]]


class PredictionOutput(BaseModel):
    predictions: List[int]
    probabilities: List[float]


@app.post("/predict", response_model=PredictionOutput)
def predict(data=None):
    if data is None:
        data = [5.1, 3.5, 1.4, 0.2]
    features = np.array(data.features)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features).max(axis=1)

    return PredictionOutput(predictions=predictions.tolist(), probabilities=probabilities.tolist())


@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}
