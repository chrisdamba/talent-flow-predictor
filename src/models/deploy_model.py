import json
import mlflow
import sagemaker
from sagemaker.mlflow import MLflowModel


def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)


def deploy_model_to_sagemaker(run_id, config):
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    model_uri = f"runs:/{run_id}/model"
    sagemaker.Session()

    mlflow_model = MLflowModel(
        model_uri=model_uri,
        role=config['sagemaker_role_arn'],
        image_uri=config.get('sagemaker_container_image_uri')
    )

    predictor = mlflow_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=config['sagemaker_endpoint_name']
    )

    print(f"Model deployed successfully to endpoint: {config['sagemaker_endpoint_name']}")
    return predictor


def test_endpoint(predictor, test_data):
    response = predictor.predict(test_data)
    print("Test prediction results:", response)


def main():
    config = load_config()
    latest_run_id = config.get('latest_mlflow_run_id')

    if not latest_run_id:
        print("No MLflow run ID found. Please provide a valid run ID.")
        return

    predictor = deploy_model_to_sagemaker(latest_run_id, config)

    test_data = [
        {"job_title": "Data Scientist", "company_name": "Tech Corp", "job_location": "New York",
         "job_skills": "Python,Machine Learning,SQL", "posting_year": 2024, "posting_month": 6,
         "job_level": "Mid-level", "skill_count": 3, "industry": "Technology", "salary_value": 100000}
    ]
    test_endpoint(predictor, test_data)


if __name__ == "__main__":
    main()
