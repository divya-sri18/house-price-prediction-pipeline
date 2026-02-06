from src.ingestion import ingest_data
train_path,test_path=ingest_data()
print(train_path)
print(test_path)