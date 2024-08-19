import json
from io import BytesIO

import mlflow
import mlflow.sklearn
import pandas as pd
import boto3

from src.data.load_data import load_data_from_s3, prepare_data
from src.models.train_model import train_model


def load_model(run_id, config):
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def make_predictions(model, X):
    return model.predict(X)


def main():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    pass


if __name__ == "__main__":
    main()
