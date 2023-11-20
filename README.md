# Wine Quality Prediction Pipeline

## Overview

This project aims to build and run a feature pipeline using Modal, train a model on the feature data, and then build an inference pipeline complete with a Gradio UI hosted on HuggingFace Spaces. The main objective is to predict the quality of wine using a set of features.

The project leverages the HopsWorks feature store for efficient data management. 

### Dataset

The dataset used in this project is the Wine Quality Dataset, which can be found [here](https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/wine.csv). It consists of various attributes that describe the wine's properties and a target quality score.

## Project Structure

- `wine-eda-and-backfilling.ipynb`: 
  - Conducts data cleaning, exploratory data analysis (EDA), and feature selection.
  - Uploads relevant data to the HopsWorks feature store.

- `wine-feature-pipeline-daily.py`: 
  - A script designed to run on Modal, generating 10 new data points daily and adding them to the feature store.

- `wine-training-pipeline.ipynb`: 
  - Handles the training process using linear regression and decision tree models.
  - Evaluates the performance of these models and uploads them to the HopsWorks model registry.

- `wine-batch-inference-pipeline.py`: 
  - Another script for Modal, fetching batch data from the feature store (the most recently added data).
  - Measures and stores the performance of the models on this batch data using the dataset_api on HopsWorks.

- `huggingface-space-wine/app.py`:
  - Contains a Gradio interface for making wine quality predictions using sliders to adjust feature values.
  - Displays the classification accuracy, confusion matrix, test MSE, and prediction comparisons of the two models on the batch data.

## How It Works

Provide a brief explanation of how your project works. For instance, you might describe the flow from data processing to training and finally to inference.

## Setup and Installation

Run the following bash script and replace the HopsWorks login api_key_value with your own api key

```bash
pip install -r requirements.txt
```

## Usage
### Perform EDA and backfilling
jupyter notebook wine-eda-and-backfilling.ipynb

### Run feature pipeline on Modal
python wine-feature-pipeline-daily.py

### Train models
jupyter notebook wine-training-pipeline.ipynb

### Execute batch inference pipeline on Modal
python wine-batch-inference-pipeline.py

### Run the Gradio interface (HuggingFace Spaces)
python huggingface-space-wine/app.py

