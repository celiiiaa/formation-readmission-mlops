from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.predict_wrapper import ReadmissionPredictor

app = FastAPI()
model = ReadmissionPredictor()


class InputData(BaseModel):
    chol: float
    crp: float
    phos: float

@app.get("/")
def root():
    return {"message": "API is ready!"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
