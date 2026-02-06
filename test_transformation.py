from src.ingestion import ingest_data
from src.transformation import transform_data
train_path,test_path=ingest_data()
X_train, X_test, y_train, y_test = transform_data(train_path, test_path)
print(X_train.shape)
print(X_test.shape)
