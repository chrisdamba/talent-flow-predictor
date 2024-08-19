# Data Preparation

This document outlines the steps to prepare the LinkedIn job listings dataset for our ML pipeline using Parquet format.

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Prerequisites](#prerequisites)
3. [Downloading the Dataset](#downloading-the-dataset)
4. [Data Extraction](#data-extraction)
5. [Data Cleaning](#data-cleaning)
6. [Feature Engineering](#feature-engineering)
7. [Data Storage](#data-storage)

## Dataset Overview

The dataset includes 1.3 million job listings collected from LinkedIn in 2024. It provides a comprehensive view of the job market, including job titles, companies, locations, required skills, and more. Our goal is to prepare this data for machine learning tasks, focusing on analyzing and predicting company hiring patterns.

## Prerequisites

- Python 3.8+
- AWS CLI configured
- Kaggle API configured
- Required Python packages:
  - pandas
  - numpy
  - boto3
  - pyarrow
  - kaggle

Install the required packages:

```bash
pip install pandas numpy boto3 pyarrow kaggle
```

## Downloading the Dataset

1. Use the Kaggle API to download the dataset:

```bash
kaggle datasets download muhammadehsan000/1-3m-linkedin-jobs-and-skills-dataset-2024
```

2. Extract the downloaded file:

```bash
unzip 1-3m-linkedin-jobs-and-skills-dataset-2024.zip
```

## Data Extraction

Use the following script to load and combine the CSV files:

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Load the CSV files
job_postings = pd.read_csv('linkedin_job_postings.csv')
job_skills = pd.read_csv('job_skills.csv')
job_summary = pd.read_csv('job_summary.csv')

# Merge the dataframes
df = job_postings.merge(job_skills, on='job_link', how='left')
df = df.merge(job_summary, on='job_link', how='left')

# Save to Parquet
table = pa.Table.from_pandas(df)
pq.write_table(table, 'linkedin_jobs_dataset.parquet')
```

Run this script to extract the data and save it as a Parquet file.

## Data Cleaning

Perform the following data cleaning steps:

```python
import pandas as pd
import pyarrow.parquet as pq

# Read the Parquet file
df = pq.read_table('linkedin_jobs_dataset.parquet').to_pandas()

# Handle missing values
numerical_cols = ['job_id', 'job_salary_from', 'job_salary_to']
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

string_cols = ['job_title', 'company_name', 'job_location', 'job_skills']
df[string_cols] = df[string_cols].fillna('Unknown')

# Remove duplicates
df.drop_duplicates(subset='job_link', keep='first', inplace=True)

# Handle outliers in salary columns
def remove_outliers(df, column, lower_percentile=0.01, upper_percentile=0.99):
    lower = df[column].quantile(lower_percentile)
    upper = df[column].quantile(upper_percentile)
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    return df

for col in ['job_salary_from', 'job_salary_to']:
    df = remove_outliers(df, col)

# Save the cleaned data
table = pa.Table.from_pandas(df)
pq.write_table(table, 'linkedin_jobs_cleaned.parquet')
```

## Feature Engineering

Create new features that might be useful for analyzing hiring patterns:

```python
import pandas as pd
import pyarrow.parquet as pq

# Read the cleaned Parquet file
df = pq.read_table('linkedin_jobs_cleaned.parquet').to_pandas()

# Create a 'job_level' feature
df['job_level'] = df['job_title'].apply(lambda x: 'Senior' if 'Senior' in x or 'Sr.' in x else 'Junior' if 'Junior' in x or 'Jr.' in x else 'Mid-level')

# Create a 'salary_range' feature
df['salary_range'] = pd.cut(df['job_salary_from'], 
                            bins=[0, 50000, 100000, 150000, float('inf')],
                            labels=['Entry', 'Mid', 'Senior', 'Executive'])

# Extract industry from job description
df['industry'] = df['job_description'].apply(lambda x: x.split(' in ')[-1].split('.')[0] if ' in ' in x else 'Unknown')

# Create a 'skills_count' feature
df['skills_count'] = df['job_skills'].apply(lambda x: len(x.split(',')))

# Save the feature-engineered data
table = pa.Table.from_pandas(df)
pq.write_table(table, 'linkedin_jobs_featured.parquet')
```

## Data Storage

Store the prepared dataset in Amazon S3:

```python
import boto3
import pyarrow.parquet as pq

# Read the Parquet file
table = pq.read_table('linkedin_jobs_featured.parquet')

# Initialize S3 client
s3 = boto3.client('s3')

# Define bucket and file names
bucket_name = 'your-s3-bucket-name'
file_name = 'linkedin_jobs_prepared.parquet'

# Upload file to S3
s3.put_object(Bucket=bucket_name, Key=file_name, Body=table.serialize().to_pybytes())

print(f"Data uploaded to S3://{bucket_name}/{file_name}")
```

Ensure you have the necessary permissions to write to the specified S3 bucket.

## Next Steps

After completing these data preparation steps, your dataset will be ready for model training. The Parquet file stored in S3 can be easily loaded and used in your ML pipeline. Proceed to the [Model Training](model_training.md) guide for the next steps in the ML pipeline.
