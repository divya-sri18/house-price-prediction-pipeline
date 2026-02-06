import joblib

preprocessor = joblib.load("models/preprocessor.pkl")

print(len(preprocessor.feature_names_in_))
print(preprocessor.feature_names_in_)
