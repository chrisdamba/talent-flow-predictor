import os
import sys

from src.data.load_data import clean_data, load_data_from_s3, engineer_features

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
import json


def load_config():
    with open('../config.json', 'r') as f:
        return json.load(f)


def save_to_s3(df, bucket, key):
    s3 = boto3.client('s3')
    table = pa.Table.from_pandas(df)
    parquet_buffer = BytesIO()
    pq.write_table(table, parquet_buffer)
    s3.put_object(Bucket=bucket, Key=key, Body=parquet_buffer.getvalue())


def main():
    config = load_config()

    print("Loading data from S3...")
    df = load_data_from_s3(config['raw_data_key'], config['s3_bucket_name'])

    print("Cleaning data...")
    df = clean_data(df)

    print("Engineering features...")
    df = engineer_features(df)

    print("Saving prepared data to S3...")
    save_to_s3(df, config['s3_bucket_name'], config['prepared_data_key'])

    print("Data preparation completed successfully!")


if __name__ == "__main__":
    main()
