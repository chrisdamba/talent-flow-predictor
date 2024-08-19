import json

import boto3
import mlflow
import pandas as pd
from mage_ai.data_preparation.decorators import data_loader, transformer

from src.data.load_data import prepare_data
from src.models.predict_model import make_predictions, recommend_songs
from src.models.train_model import train_model


@data_loader
def load_data_from_s3(*args, **kwargs):
    """
    Load data from S3 using the configuration
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    pass


@transformer
def train_and_evaluate_model(data, *args, **kwargs):
    """
    Train and evaluate the model
    """
    pass


@transformer
def generate_recommendations(run_id, *args, **kwargs):
    """
    Generate recommendations using the trained model
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    pass


@transformer
def save_recommendations(recommendations, *args, **kwargs):
    """
    Save recommendations to S3
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    output_key = "recommendations.csv"
    s3 = boto3.client('s3')
    csv_buffer = pd.DataFrame({'recommendations': recommendations}).to_csv(index=False)
    s3.put_object(Bucket=config['s3_bucket_name'], Key=output_key, Body=csv_buffer)

    return f"Recommendations saved to S3://{config['s3_bucket_name']}/{output_key}"
