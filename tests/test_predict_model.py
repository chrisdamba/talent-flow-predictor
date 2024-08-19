import json

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.models.predict_model import load_model, make_predictions, main


@pytest.fixture
def mock_config():
    return {
        'mlflow_tracking_uri': 'http://mock-mlflow-server',
        's3_bucket_name': 'mock-bucket',
        's3_key_name': 'mock-data.parquet',
        'latest_mlflow_run_id': 'mock_run_id'
    }


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = [100000, 120000, 90000]
    return model


def test_load_model(mock_config):
    with patch('mlflow.sklearn.load_model') as mock_load:
        mock_load.return_value = 'mock_model'
        model = load_model('mock_run_id', mock_config)
        assert model == 'mock_model'
        mock_load.assert_called_once_with('runs:/mock_run_id/model')


def test_make_predictions(mock_model):
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    predictions = make_predictions(mock_model, X)
    assert len(predictions) == 3
    mock_model.predict.assert_called_once_with(X)


@patch('src.data.load_data.prepare_data')
@patch('src.models.predict_model.load_model')
@patch('pandas.DataFrame.to_csv')
@patch('boto3.client')
def test_main(mock_boto3, mock_to_csv, mock_load_model, mock_prepare, mock_config):
    # Mock the S3 client and its operations
    mock_s3 = MagicMock()
    mock_boto3.return_value = mock_s3
    mock_s3.get_object.return_value = {'Body': MagicMock()}

    # Mock the prepare_data function
    mock_prepare.return_value = (pd.DataFrame({'feature1': [1, 2, 3]}), pd.Series([100, 200, 300]), ['feature1'])

    # Mock the load_model function
    mock_model = MagicMock()
    mock_model.predict.return_value = [150, 250, 350]
    mock_load_model.return_value = mock_model

    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_config)
        main()

    mock_prepare.assert_called_once_with(mock_config)
    mock_load_model.assert_called_once_with('mock_run_id', mock_config)
    mock_to_csv.assert_called_once()
    mock_boto3.assert_called_once_with('s3')
    mock_s3.get_object.assert_called_once_with(Bucket='mock-bucket', Key='mock-data.parquet')
