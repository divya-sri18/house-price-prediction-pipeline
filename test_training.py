from src.ingestion import ingest_data
from src.transformation import transform_data
from src.model_trainer import train_model

train_path,test_path=ingest_data()
X_train,X_test,y_train,y_test=transform_data(train_path,test_path)
model=train_model(X_train,y_train)

print("Model trained and saved.")