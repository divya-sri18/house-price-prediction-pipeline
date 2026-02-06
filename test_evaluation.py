from src.ingestion import ingest_data
from src.transformation import transform_data
from src.model_trainer import train_model
from src.evaluation import evaluate_model

train_path, test_path = ingest_data()

X_train, X_test, y_train, y_test = transform_data(train_path, test_path)

model = train_model(X_train, y_train)

rmse, r2 = evaluate_model(model, X_test, y_test)

print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")
