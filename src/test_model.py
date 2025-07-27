# src/test_model.py
import joblib
import pandas as pd

def test_model():
    try:
        model = joblib.load("models/model.joblib")
        df = pd.DataFrame([[1, 2, 3, 4]])
        prediction = model.predict(df)
        print("✅ Prediction successful:", prediction)
        assert prediction is not None
    except Exception as e:
        print("❌ Test failed:", e)
        raise

if __name__ == "__main__":
    test_model()