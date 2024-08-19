import json
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.data.load_data import prepare_data


def train_model(X, y, feature_names, popular_songs) -> str:
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    mlflow.set_experiment("talent_flow_prediction")
    # Train the model
    with mlflow.start_run():
        # Split the data
        pass

    return mlflow.active_run().info.run_id


if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    pass
