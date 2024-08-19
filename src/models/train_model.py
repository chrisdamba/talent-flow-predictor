import json
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.data.load_data import prepare_data


def train_model(X, y, config):
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    mlflow.set_experiment("talent_flow_prediction")

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained. MSE: {mse}, R2: {r2}")
        return mlflow.active_run().info.run_id


if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    X, y, features = prepare_data(config)
    run_id = train_model(X, y, config)
    print(f"Model training completed. Run ID: {run_id}")
