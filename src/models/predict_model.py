import json
import mlflow
import mlflow.sklearn
import pandas as pd

from src.data.load_data import prepare_data


def load_model(run_id, config):
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def make_predictions(model, X):
    return model.predict(X)


def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    X, y, features = prepare_data(config)

    run_id = config.get('latest_mlflow_run_id')
    model = load_model(run_id, config)

    predictions = make_predictions(model, X)

    results = pd.DataFrame({'actual': y, 'predicted': predictions})
    results.to_csv('hiring_trend_predictions.csv', index=False)
    print("Predictions saved to hiring_trend_predictions.csv")


if __name__ == "__main__":
    main()
