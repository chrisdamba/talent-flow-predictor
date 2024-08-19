from io import BytesIO
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data.load_data import clean_data, engineer_features, load_data_from_s3, prepare_data


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'job_title': ['Data Scientist', 'Software Engineer', np.nan],
        'company_name': ['Tech Corp', np.nan, 'AI Inc'],
        'job_location': ['New York', 'San Francisco', np.nan],
        'job_skills': ['Python,SQL', np.nan, 'Java,C++'],
        'job_salary': ['$100000', '€80000', np.nan],
        'date_posted': ['2024-01-01', '2024-02-01', '2024-03-01'],
        'job_description': ['Data science role in tech', 'Software development in finance', 'AI research']
    })


def test_clean_data(sample_df):
    cleaned_df = clean_data(sample_df)
    assert cleaned_df['job_title'].isnull().sum() == 0
    assert cleaned_df['company_name'].isnull().sum() == 0
    assert cleaned_df['job_location'].isnull().sum() == 0
    assert cleaned_df['job_skills'].isnull().sum() == 0
    assert 'salary_currency' in cleaned_df.columns
    assert 'salary_value' in cleaned_df.columns
    assert len(cleaned_df) == len(sample_df)


def test_engineer_features(sample_df):
    engineered_df = engineer_features(sample_df)
    assert 'posting_year' in engineered_df.columns
    assert 'posting_month' in engineered_df.columns
    assert 'job_level' in engineered_df.columns
    assert 'skill_count' in engineered_df.columns
    assert 'industry' in engineered_df.columns
    assert engineered_df['posting_year'].dtype == np.dtype('int64')
    assert engineered_df['posting_month'].dtype == np.dtype('int64')
    assert engineered_df['job_level'].dtype == 'object'
    assert engineered_df['skill_count'].dtype == np.dtype('int64')
    assert engineered_df['industry'].dtype == 'object'
    assert 'Unknown' in engineered_df['job_level'].values
    assert engineered_df['skill_count'].min() == 1


@patch('boto3.client')
def test_load_data_from_s3(mock_boto3):
    mock_s3 = MagicMock()
    mock_boto3.return_value = mock_s3
    mock_body = MagicMock()
    mock_body.read.return_value = b'mock parquet data'
    mock_s3.get_object.return_value = {'Body': mock_body}

    with patch('pyarrow.parquet.read_table') as mock_read_table:
        mock_read_table.return_value.to_pandas.return_value = pd.DataFrame({'col1': [1, 2, 3]})
        result = load_data_from_s3('mock_key', 'mock_bucket')

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    mock_s3.get_object.assert_called_once_with(Bucket='mock_bucket', Key='mock_key')
    mock_body.read.assert_called_once()
    mock_read_table.assert_called_once()

    # Check that read_table was called with a BytesIO object containing the correct data
    args, kwargs = mock_read_table.call_args
    assert len(args) == 1
    assert isinstance(args[0], BytesIO)
    assert args[0].getvalue() == b'mock parquet data'


@patch('src.data.load_data.load_data_from_s3')
@patch('src.data.load_data.clean_data')
@patch('src.data.load_data.engineer_features')
def test_prepare_data(mock_engineer, mock_clean, mock_load, sample_df):
    mock_load.return_value = sample_df
    mock_clean.return_value = sample_df

    # Create a DataFrame that includes all the expected engineered features
    engineered_df = sample_df.copy()
    engineered_df['posting_year'] = 2024
    engineered_df['posting_month'] = 1
    engineered_df['job_level'] = 'Mid-level'
    engineered_df['skill_count'] = 2
    engineered_df['industry'] = 'Tech'
    engineered_df['salary_value'] = 100000

    mock_engineer.return_value = engineered_df

    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = ('{"s3_key_name": "mock_key", '
                                                                           '"s3_bucket_name": "mock_bucket"}')
        X, y, features = prepare_data({'s3_key_name': 'mock_key', 's3_bucket_name': 'mock_bucket'})

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(features, list)
    assert all(feature in engineered_df.columns for feature in features)
    assert 'salary_value' not in features
    assert y.name == 'salary_value'
    assert set(features) == {'job_title', 'company_name', 'job_location', 'job_skills', 'posting_year', 'posting_month',
                             'job_level', 'skill_count', 'industry'}

    mock_load.assert_called_once()
    mock_clean.assert_called_once()
    mock_engineer.assert_called_once()
