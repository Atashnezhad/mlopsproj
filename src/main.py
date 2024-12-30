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
    predictions: int
    probabilities: float


@app.post("/predict", response_model=PredictionOutput)
def predict(data: InputData):

    features = np.array(data.features)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features).max(axis=1)

    return PredictionOutput(predictions=predictions, probabilities=probabilities)


@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}
