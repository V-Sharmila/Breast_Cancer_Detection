# Breast Cancer Diagnosis Classification Using AdaBoost

This project implements a classification model using **AdaBoost** for diagnosing breast cancer as benign or malignant based on the features from the Wisconsin Breast Cancer dataset. The dataset contains various features related to breast cell characteristics, and the model is built to predict whether the tumor is malignant or benign.

The project involves several stages including data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, performance evaluation, and visualization of results.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [Steps](#steps)
- [Results](#results)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Running the Streamlit App](#running-the-streamlit-app)

## Project Overview

In this project, we used the **Wisconsin Breast Cancer dataset** from the UCI Machine Learning Repository. We applied AdaBoost (Adaptive Boosting) with a base estimator of Decision Tree Classifier to classify tumors as benign or malignant. We also fine-tuned the model using **GridSearchCV** to optimize hyperparameters and evaluate performance.

The project steps include:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Model training with AdaBoost
- Hyperparameter tuning
- Performance evaluation using various metrics like accuracy, confusion matrix, ROC curve, and learning curve
- Feature importance visualization

## Dataset

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**. It contains 569 samples with 30 features representing various tumor characteristics, along with the diagnosis ('M' for malignant and 'B' for benign). 

You can access the dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)).

### Columns:
- `id`: Unique identifier for each sample (dropped for the analysis).
- `diagnosis`: Target variable, indicating if the tumor is benign ('B') or malignant ('M').
- `feature_1` to `feature_30`: Features representing various tumor characteristics.

## Dependencies

To run this project, you'll need the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`
- `imblearn`
- `streamlit`

## File Structure
BreastCancerDiagnosis/
├── README.md                # Project documentation
├── breast_cancer_classifier.ipynb  # Jupyter notebook for ML workflow
├── app.py                   # Streamlit application for interactive predictions
└── data/
    └── breast_cancer.csv    # Dataset used in the project
    
## Steps
1.Data Preprocessing
2.EDA
3.Model training
4.Hyperparameetr Tuning
5.Performance Evaluation
6.Feature Importance
7.Streamlit Web APP

## Results
Accuracy: The final model achieved a high accuracy (around 98%) on the test data, meaning it correctly classified most of the samples.
Confusion Matrix: The model demonstrated a low number of false positives and false negatives.
ROC Curve: The ROC curve was close to the top-left corner, indicating excellent classification performance.
Learning Curve: The learning curve showed that the model's accuracy improved significantly as more training data was used.

## Hyperparameter Tuning
Best Hyperparameters: The GridSearchCV returned the best combination of hyperparameters:
->n_estimators = 100
->learning_rate = 1.0 These hyperparameters provided the best performance on the test set, ensuring both good bias and low variance.

## Evaluation Metrics
Accuracy: Accuracy was used to measure the percentage of correct predictions.
Precision: Precision was calculated to understand how many of the predicted malignant tumors were truly malignant.
Recall: Recall was calculated to assess how many of the actual malignant tumors were correctly identified.
F1-Score: The F1-score balanced precision and recall, providing an overall performance measure.
ROC Curve: The ROC curve and the AUC value confirmed the model’s ability to differentiate between benign and malignant tumors.
![image](https://github.com/user-attachments/assets/39b66f67-5646-4519-be6b-b64614c48e30)
![image](https://github.com/user-attachments/assets/e4ab0146-0cb2-42fa-ac10-d5d6c68e9f23)

## Feature Importance
![image](https://github.com/user-attachments/assets/2ab25ff5-1cbb-4b53-ac7b-57e0ff46ecee)

## Running the Streamlit App
1.First, install the necessary dependencies by running:
pip install -r requirements.txt
2.Launch the Streamlit app with the following command:
streamlit run app.py

Open the app in a browser. You can input tumor feature values and get an instant prediction for benign or malignant.

![image](https://github.com/user-attachments/assets/b5c331ac-58ca-4fb3-9799-2fa645425992)

![image](https://github.com/user-attachments/assets/4aaa11bd-14bf-4a00-b2c9-4025dd264e3f)




