import json
import boto3
import mlflow
import pandas as pd
from mage_ai.data_preparation.decorators import data_loader, transformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.data.load_data import prepare_data


@data_loader
def load_data(*args, **kwargs):
    with open('config.json', 'r') as f:
        config = json.load(f)

    X, y, features = prepare_data(config)
    return X, y, features


@transformer
def train_and_evaluate_model(data, *args, **kwargs):
    X, y, features = data
    with open('config.json', 'r') as f:
        config = json.load(f)

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


@transformer
def generate_predictions(run_id, *args, **kwargs):
    with open('config.json', 'r') as f:
        config = json.load(f)

    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    X, _, _ = load_data()
    predictions = model.predict(X)
    return predictions


@transformer
def save_predictions(predictions, *args, **kwargs):
    with open('config.json', 'r') as f:
        config = json.load(f)

    output_key = "hiring_trend_predictions.csv"
    s3 = boto3.client('s3')
    csv_buffer = pd.DataFrame({'predictions': predictions}).to_csv(index=False)
    s3.put_object(Bucket=config['s3_bucket_name'], Key=output_key, Body=csv_buffer)

    return f"Predictions saved to S3://{config['s3_bucket_name']}/{output_key}"
