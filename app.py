from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

model = joblib.load("models/model.joblib")

@app.get("/")
def read_root():
    return {"message": "ML model is live 🚀"}

@app.post("/predict")
def predict(data: InputData):
    features = [[
        data.feature1, data.feature2, data.feature3, data.feature4
    ]]
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}


