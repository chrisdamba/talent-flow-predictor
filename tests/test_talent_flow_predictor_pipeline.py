import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.mage_ai_pipelines.talent_flow_predictor_pipeline import load_data, train_and_evaluate_model, \
    generate_predictions, save_predictions


@pytest.fixture
def mock_config():
    return {
        's3_bucket_name': 'mock-bucket',
        's3_key_name': 'mock-data.parquet',
        'mlflow_tracking_uri': 'http://mock-mlflow-server'
    }


@patch('src.data.load_data.prepare_data')
@patch('builtins.open')
def test_load_data(mock_open, mock_prepare_data, mock_config):
    # Mock the file opening operation
    mock_open.return_value.__enter__.return_value.read.return_value = ('{"s3_bucket_name": "mock-bucket", '
                                                                       '"s3_key_name": "mock-data.parquet"}')

    # Mock the prepare_data function
    mock_X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    mock_y = pd.Series([100, 200, 300])
    mock_features = ['feature1', 'feature2']
    mock_prepare_data.return_value = (mock_X, mock_y, mock_features)

    result = load_data()

    assert result == (mock_X, mock_y, mock_features)
    mock_prepare_data.assert_called_once()
    mock_open.assert_called_once_with('config.json', 'r')


@patch('src.models.train_model.train_model')
@patch('builtins.open')
def test_train_and_evaluate_model(mock_open, mock_train_model, mock_config):
    # Mock the file opening operation
    mock_open.return_value.__enter__.return_value.read.return_value = ('{"mlflow_tracking_uri": '
                                                                       '"http://mock-mlflow-server"}')

    # Mock the train_model function
    mock_train_model.return_value = 'mock_run_id'

    # Create mock data
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([100, 200, 300])
    features = ['feature1', 'feature2']

    data = (X, y, features)

    result = train_and_evaluate_model(data)

    assert result == 'mock_run_id'
    mock_train_model.assert_called_once_with(X, y, features)
    mock_open.assert_called_once_with('config.json', 'r')


@patch('mlflow.sklearn.load_model')
@patch('src.models.predict_model.make_predictions')
@patch('src.mage_ai_pipelines.talent_flow_predictor_pipeline.load_data')
def test_generate_predictions(mock_load_data, mock_predict, mock_load_model, mock_config):
    mock_load_model.return_value = MagicMock()
    mock_load_data.return_value = ('X', 'y', 'features')
    mock_predict.return_value = 'predictions'

    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = '{"key": "value"}'
        result = generate_predictions('mock_run_id')

    assert result == 'predictions'
    mock_load_model.assert_called_once_with('runs:/mock_run_id/model')
    mock_load_data.assert_called_once()
    mock_predict.assert_called_once()


@patch('boto3.client')
def test_save_predictions(mock_boto3, mock_config):
    predictions = [100000, 120000, 90000]

    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = '{"key": "value"}'
        result = save_predictions(predictions)

    assert 'Predictions saved to S3' in result
    mock_boto3.assert_called_once_with('s3')
    mock_boto3.return_value.put_object.assert_called_once()
