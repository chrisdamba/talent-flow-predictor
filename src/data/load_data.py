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

    # Remove duplicates (using all columns since we don't have a specific 'job_link')
    df.drop_duplicates(keep='first', inplace=True)

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
    def get_job_level(title):
        if pd.isna(title):
            return 'Unknown'
        elif 'Senior' in title or 'Sr.' in title:
            return 'Senior'
        elif 'Junior' in title or 'Jr.' in title:
            return 'Junior'
        else:
            return 'Mid-level'

    df['job_level'] = df['job_title'].apply(get_job_level)

    # Create a skill count feature
    df['skill_count'] = df['job_skills'].fillna('').str.count(',') + 1

    # Create an industry feature (this is a simplification, you might want to use a more sophisticated method)
    df['industry'] = df['job_description'].apply(
        lambda x: x.split(' in ')[-1].split('.')[0] if pd.notna(x) and ' in ' in x else 'Unknown')

    return df


def prepare_data(config: dict) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare the dataset for machine learning."""
    # Load data from S3
    df = load_data_from_s3(config['s3_key_name'], config['s3_bucket_name'])

    # Clean the data
    df = clean_data(df)

    # Engineer features
    df = engineer_features(df)

    # Select features for the model
    features = ['job_title', 'company_name', 'job_location', 'job_skills', 'posting_year',
                'posting_month', 'job_level', 'skill_count', 'industry']

    # Only use features that are actually present in the DataFrame
    available_features = [f for f in features if f in df.columns]

    # Prepare X and y
    X = df[available_features]
    y = df['salary_value'] if 'salary_value' in df.columns else pd.Series(dtype='float64')

    return X, y, available_features


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
