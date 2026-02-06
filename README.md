# House Price Prediction — End-to-End Machine Learning Pipeline

This project implements a complete **end-to-end machine learning pipeline** for predicting house prices using structured housing data.  
The focus is not just on model accuracy, but on building a **production-style ML workflow** with clean structure, reproducibility, and separation of concerns.

## Problem Statement

Given historical housing data with multiple numerical and categorical features, the goal is to predict the **sale price of a house**.

This project treats the problem the way it would be handled in a real ML system:
- Raw data ingestion
- Feature preprocessing using pipelines
- Model training and evaluation
- Saving trained artifacts for reuse
- Manual inference workflow

## Project Structure

house-price-prediction-pipeline/
├── data/
│ ├── raw/
│ └── processed/
├── src/
│ ├── ingestion.py
│ ├── transformation.py
│ └── training.py
├── models/
│ ├── model.pkl
│ └── preprocessor.pkl
├── test_ingestion.py
├── test_transformation.py
├── test_training.py
└── requirements.txt


## Dataset

- Ames Housing Dataset  
- Target variable: `SalePrice`

## Model

- Algorithm: Random Forest Regressor
- Metrics:
  - R² ≈ 0.91
  - RMSE ≈ 26,750

## How to Run
pip install -r requirements.txt
python test_ingestion.py
python test_transformation.py
python test_training.py

