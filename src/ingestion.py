import pandas as pd
from sklearn.model_selection import train_test_split
import os

def ingest_data():
    df=pd.read_csv(r"C:\Users\Lenovo\Desktop\house-price-prediction\data\raw\AmesHousing.csv"
)
    train_df,test_df=train_test_split(df,test_size=0.2,random_state=42)
    os.makedirs("data/processed", exist_ok=True)

    
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path