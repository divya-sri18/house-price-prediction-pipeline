import pandas as pd

df = pd.read_csv("data/raw/AmesHousing.csv")

sample_input = df.drop(columns=["SalePrice"]).iloc[0].to_dict()
