import joblib
import pandas as pd

model = joblib.load("models/model.joblib")

def predict(input_dict):
    df = pd.DataFrame([input_dict])
    return model.predict(df)[0]
