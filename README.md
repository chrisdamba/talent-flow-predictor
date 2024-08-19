# TalentFlowPredictor: Company Hiring Pattern Analysis

An end-to-end Machine Learning Operations (MLOps) project that uses LinkedIn job listings data to analyze and predict company hiring patterns using AWS, Terraform, Docker, MLflow, and more!

## Description

### Objective

This project aims to build a comprehensive MLOps pipeline for analyzing and predicting company hiring patterns using LinkedIn job listings data. We will develop machine learning models to identify trends in hiring across different companies, industries, and locations. The pipeline will cover the entire ML lifecycle, including data preparation, model training, experiment tracking, model deployment, and monitoring.

### Dataset

We're using a dataset of 1.3 million job listings collected from LinkedIn in 2024. This dataset provides rich information about job postings, including job titles, companies, locations, required skills, and more.

### Tools & Technologies

- Cloud Provider - [**Amazon Web Services (AWS)**](https://aws.amazon.com)
- Infrastructure as Code - [**Terraform**](https://www.terraform.io)
- Containerization - [**Docker**](https://www.docker.com), [**Docker Compose**](https://docs.docker.com/compose/)
- ML Framework - [**Scikit-learn**](https://scikit-learn.org/)
- Experiment Tracking and Model Registry - [**MLflow**](https://mlflow.org/)
- Workflow Orchestration - [**Apache Airflow**](https://airflow.apache.org/)
- Model Deployment - [**AWS SageMaker**](https://aws.amazon.com/sagemaker/)
- Model Monitoring - [**Amazon CloudWatch**](https://aws.amazon.com/cloudwatch/)
- Data Storage - [**Amazon S3**](https://aws.amazon.com/s3/) (Data Lake), [**Amazon RDS**](https://aws.amazon.com/rds/) (Metadata)
- CI/CD - [**GitHub Actions**](https://github.com/features/actions)
- Language - [**Python**](https://www.python.org)

### Architecture

![mlops-architecture](images/MLOPs_Architecture.jpg)

## Setup

**WARNING: You will be charged for the AWS resources created in this project. You can use the AWS Free Tier for some services, but not all.**

### Prerequisites

- AWS Account and configured AWS CLI
- Terraform installed
- Docker and Docker Compose installed
- Python 3.8+

### Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/chrisdamba/talent-flow-predictor.git
   cd talent-flow-predictor
   ```

2. Set up the infrastructure with Terraform:
   ```
   cd terraform
   terraform init
   terraform apply
   ```

3. Set up the development environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

4. Follow the setup instructions in the `setup/` directory for each component:
   - [Data Preparation](setup/data_prep.md)
   - [Model Training](setup/model_training.md)
   - [MLflow Setup](setup/mlflow.md)
   - [Airflow Workflow](setup/airflow.md)
   - [Model Deployment](setup/deployment.md)
   - [Monitoring](setup/monitoring.md)

## Project Structure

```
talent-flow-predictor/
├── data/
├── models/
├── notebooks/
├── scripts/
│   ├── data_prep.py
│   ├── train_model.py
│   └── deploy_model.py
├── src/
│   ├── features/
│   ├── models/
│   └── utils/
├── tests/
├── setup/
│   ├── data_prep.md
│   ├── model_training.md
│   ├── mlflow.md
│   ├── airflow.md
│   ├── deployment.md
│   └── monitoring.md
├── terraform/
├── .github/workflows/ci_cd.yml
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

## Usage

This section provides instructions on how to use the TalentFlowPredictor project, including running experiments, deploying models, and monitoring performance.

### Setting Up the Environment

1. Clone the repository:
   ```
   git clone https://github.com/chrisdamba/talent-flow-predictor.git
   cd talent-flow-predictor
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Data Preparation

1. Navigate to the scripts directory:
   ```
   cd scripts
   ```

2. Run the data preparation script:
   ```
   python data_prep.py
   ```
   This script will process the LinkedIn job listings data and store the prepared data in your S3 bucket.

### Running Experiments

1. Start the MLflow tracking server:
   ```
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 --port 5000
   ```

2. In a new terminal, run the model training script:
   ```
   python train_model.py
   ```
   This script will train the model to predict hiring patterns and log metrics and artifacts to MLflow.

3. View the MLflow UI by navigating to `http://localhost:5000` in your web browser.

### Deploying Models

1. Choose the best performing model from the MLflow UI.

2. Update the `model_uri` in the `deploy_model.py` script with the URI of your chosen model.

3. Run the deployment script:
   ```
   python deploy_model.py
   ```
   This will deploy your model to AWS SageMaker.

### Making Predictions

Once your model is deployed, you can make predictions using the AWS SageMaker Runtime:

```python
import boto3
import json

client = boto3.client('sagemaker-runtime')

# Prepare your input data
input_data = {...}  # Your input data in the correct format

response = client.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='application/json',
    Body=json.dumps(input_data)
)

result = json.loads(response['Body'].read().decode())
print(result)
```

### Monitoring Model Performance

1. Navigate to the AWS CloudWatch console.

2. Find the logs for your SageMaker endpoint.

3. Set up CloudWatch Alarms for metrics like model latency and error rates.

### Running the Full Pipeline

To run the entire pipeline from data preparation to model deployment:

1. Ensure you have set up the necessary AWS credentials and permissions.

2. Navigate to the project root directory.

3. Start the Airflow webserver and scheduler:
   ```
   airflow webserver -p 8080
   airflow scheduler
   ```

4. Access the Airflow UI at `http://localhost:8080` and trigger the `talent_flow_predictor_pipeline` DAG.

### Continuous Integration/Continuous Deployment (CI/CD)

The project uses GitHub Actions for CI/CD. On every push to the main branch:

1. Tests are automatically run.
2. If tests pass, the model is retrained on the latest data.
3. If the new model performs better than the currently deployed model, it is automatically deployed to production.

You can view the status of these workflows in the "Actions" tab of the GitHub repository.

## Development

This project uses [pre-commit](https://pre-commit.com/) to enforce coding standards and catch common issues before they are committed.

To set up pre-commit:

1. Install pre-commit:
   ```
   pip install pre-commit
   ```

2. Set up the git hooks:
   ```
   pre-commit install
   ```

Now, pre-commit will run automatically on `git commit`.

To run pre-commit manually on all files:
   ```
   pre-commit run --all-files
   ```

## Acknowledgements

- Kaggle LinkedIn Dataset](https://www.kaggle.com/datasets/muhammadehsan000/1-3m-linkedin-jobs-and-skills-dataset-2024)
- [DataTalks.Club](https://datatalks.club) for inspiration and learning resources
