from io import BytesIO

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data.load_data import clean_data, engineer_features, load_data_from_s3, prepare_data


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': ['1', '2', '3'],
    })


def test_clean_data(sample_df):
    cleaned_df = clean_data(sample_df)
    assert len(cleaned_df) == len(sample_df)

