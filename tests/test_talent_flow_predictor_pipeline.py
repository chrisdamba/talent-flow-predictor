import json

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, mock_open
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
@patch('boto3.client')
def test_load_data(mock_boto3, mock_prepare_data, mock_config):
    mock_s3 = MagicMock()
    mock_boto3.return_value = mock_s3
    mock_s3.get_object.return_value = {'Body': MagicMock(read=lambda: b'mock data')}

    mock_X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    mock_y = pd.Series([100, 200, 300])
    mock_features = ['feature1', 'feature2']
    mock_prepare_data.return_value = (mock_X, mock_y, mock_features)

    with patch('builtins.open', new_callable=mock_open, read_data=json.dumps(mock_config)):
        result = load_data()

    assert result == (mock_X, mock_y, mock_features)
    mock_prepare_data.assert_called_once()


@patch('src.models.train_model.train_model')
def test_train_and_evaluate_model(mock_train_model, mock_config):
    mock_train_model.return_value = 'mock_run_id'

    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([100, 200, 300])
    features = ['feature1', 'feature2']

    data = (X, y, features)

    with patch('builtins.open', new_callable=mock_open, read_data=json.dumps(mock_config)):
        result = train_and_evaluate_model(data)

    assert result == 'mock_run_id'
    mock_train_model.assert_called_once_with(X, y, features)


@patch('mlflow.sklearn.load_model')
@patch('src.models.predict_model.make_predictions')
@patch('src.mage_ai_pipelines.talent_flow_predictor_pipeline.load_data')
def test_generate_predictions(mock_load_data, mock_predict, mock_load_model, mock_config):
    mock_load_model.return_value = MagicMock()
    mock_load_data.return_value = (pd.DataFrame({'feature1': [1, 2, 3]}), None, None)
    mock_predict.return_value = [150, 250, 350]

    with patch('builtins.open', new_callable=mock_open, read_data=json.dumps(mock_config)):
        result = generate_predictions('mock_run_id')

    assert result == [150, 250, 350]
    mock_load_model.assert_called_once()
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
