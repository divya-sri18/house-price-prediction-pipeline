import joblib
import pandas as pd

def predict_price(input_data: dict):

    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("models/model.pkl")

    input_df = pd.DataFrame([input_data])

    processed_data = preprocessor.transform(input_df)

    prediction = model.predict(processed_data)

    return prediction[0]
