import joblib
from sklearn.ensemble import RandomForestRegressor
def train_model(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")

    return model
