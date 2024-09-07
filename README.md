# Project Analysis and Documentation

## Overview

This project is designed to preprocess, train, and evaluate a text classification model for emotion detection. The workflow includes loading datasets, cleaning and preprocessing data, tokenizing text, training a model, and evaluating its performance.

The project directory includes:
- `dataset` directory containing the CSV file with data.
- Python scripts for the pipeline and model training.
- Jupyter notebooks for experimentation and code execution.
- `requirements.txt` file listing the necessary libraries.

## Requirements

The required Python libraries are listed in the `requirements.txt` file:

**Make sure you use python 3.9**

- `datasets==2.21.0`
- `scikit-learn==1.5.1`
- `transformers==4.44.2`
- `torch==2.4.1`
- `pandas==2.2.2`


## Pipeline Class: `Piplines_Prepare_preprocessing_training`

#### Initialization

The class is initialized with column names for input text and target emotion:

- `input_column_name`: Name of the column containing text data.
- `traget_column_name`: Name of the column containing emotion labels.

#### Methods

- `data_importer`: Loads the dataset from a CSV file and prints the first few rows for inspection.

- `data_cleaner`: Cleans the dataset by:
  - Converting text to lowercase.
  - Removing special characters, numbers, emojis, and extra spaces.
  - Filtering out rare words and empty strings.

- `data_preprocessor`: Tokenizes the text data and prepares datasets for training, validation, and testing. Encodes labels and saves the label encoder object.

- `model_trainer`: Sets up training parameters and trains the model using the `Trainer` class. Saves the trained model.

- `model_evaluation`: Evaluates the trained model on the test dataset and prints the evaluation results.

- `run_pipline`: Orchestrates the entire workflow from data import to model evaluation.

## Execution Script

The execution script initializes the model and tokenizer, and runs the pipeline. For inference with a fine-tuned model, it involves:
- Loading the fine-tuned model.
- Performing inference on new text data.

## Dataset Information

The dataset used has the following classes and their counts:

- **Happy**: 7029 samples
- **Sadness**: 6265 samples
- **Anger**: 2993 samples
- **Fear**: 2652 samples
- **Love**: 1641 samples
- **Surprise**: 879 samples

## Summary

The project involves a comprehensive pipeline for emotion detection using text data. It includes:

- **Data Import**: Loading and inspecting the dataset.
- **Data Cleaning**: Preprocessing text data through various cleaning steps.
- **Data Preparation**: Tokenizing and preparing the dataset for training and evaluation.
- **Model Training**: Training a classification model using specified training parameters.
- **Model Evaluation**: Assessing the performance of the trained model on a test set.

The combination of preprocessing with pandas and scikit-learn, and model training with transformers, results in a robust emotion classification system.
