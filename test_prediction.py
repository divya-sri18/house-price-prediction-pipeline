from src.predict import predict_price
from config.sample_input import sample_input

price = predict_price(sample_input)
print("Predicted Price:", price)


