from __future__ import annotations

import json
from io import BytesIO
from typing import Tuple

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_data_from_s3(file_key: str, bucket_name: str) -> pd.DataFrame:
    """Load a Parquet file from S3 and return as a pandas DataFrame."""
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    parquet_file = BytesIO(obj['Body'].read())
    return pq.read_table(parquet_file).to_pandas()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the LinkedIn job listings dataset."""
    # Handle missing values
    df['job_title'] = df['job_title'].fillna('Unknown')
    df['company_name'] = df['company_name'].fillna('Unknown')
    df['job_location'] = df['job_location'].fillna('Unknown')
    df['job_skills'] = df['job_skills'].fillna('')

    # Remove duplicates
    df.drop_duplicates(subset=['job_link'], keep='first', inplace=True)

    # Convert salary to numeric and handle currency
    df['salary_currency'] = df['job_salary'].str.extract(r'(\$|€|£)')
    df['salary_value'] = df['job_salary'].str.extract(r'(\d+)').astype(float)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for the hiring trend analysis."""
    # Extract year and month from date_posted
    df['date_posted'] = pd.to_datetime(df['date_posted'])
    df['posting_year'] = df['date_posted'].dt.year
    df['posting_month'] = df['date_posted'].dt.month

    # Create a job level feature
    df['job_level'] = df['job_title'].apply(lambda x: 'Senior' if 'Senior' in x or 'Sr.' in x
                                            else 'Junior' if 'Junior' in x or 'Jr.' in x
                                            else 'Mid-level')

    # Create a skill count feature
    df['skill_count'] = df['job_skills'].str.count(',') + 1

    # Create an industry feature (this is a simplification, you might want to use a more sophisticated method)
    df['industry'] = df['job_description'].apply(lambda x: x.split(' in ')[-1].split('.')[0] if ' in ' in x else 'Unknown')

    return df


def prepare_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Prepare the dataset for machine learning."""
    # Load data from S3
    df = load_data_from_s3(config['s3_key_name'], config['s3_bucket_name'])

    # Clean the data
    df = clean_data(df)

    # Engineer features
    df = engineer_features(df)

    # Select features for the model
    features = ['job_title', 'company_name', 'job_location', 'job_skills', 'posting_year', 'posting_month',
                'job_level', 'skill_count', 'industry', 'salary_value']

    # Prepare X and y
    X = df[features]
    y = df['salary_value']  # Using salary as a proxy for hiring trend, adjust as needed

    return X, y, features


if __name__ == "__main__":
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Prepare the data
    X, y, features = prepare_data(config)

    print(f"Data prepared. Shape of X: {X.shape}, Shape of y: {y.shape}")
    print(f"Features used: {features}")

    # Optionally, save the prepared data
    output_file = 'prepared_data.parquet'
    pq.write_table(pa.Table.from_pandas(X), output_file)
    print(f"Prepared data saved to {output_file}")