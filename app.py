from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict

app = FastAPI()

class Features(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.post("/predict")
def get_prediction(features: Features):
    result = predict(features.dict())
    return {"prediction": int(result)}  # 👈 Convert to int here

