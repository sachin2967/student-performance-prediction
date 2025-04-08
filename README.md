# Student Performance Prediction

## Description
The **Student Performance Prediction** project aims to predict whether students will pass or fail based on their academic, social, and demographic attributes. Using data from Portuguese schools, this project applies machine learning techniques such as Random Forest Classifier to classify students into binary categories ("pass" or "fail") based on their final grades (`G3`). The pipeline includes data preprocessing, exploratory data analysis (EDA), feature engineering, and model training. The project achieves high accuracy in predictions and provides insights into factors affecting student performance.

---

## Features
- **Data Preprocessing**:
  - Handles missing values for numerical and categorical variables.
  - Encodes binary categorical variables using Label Encoding.
  - Encodes nominal categorical variables using One-Hot Encoding.
  - Normalizes numerical features using StandardScaler.
- **Exploratory Data Analysis (EDA)**:
  - Visualizes the distribution of grades (`G3`) and other key features.
  - Summarizes data statistics and identifies missing values.
- **Machine Learning Model**:
  - Implements a Random Forest Classifier for binary classification.
  - Achieves high accuracy (96.17%) in predicting student performance.
- **Evaluation Metrics**:
  - Provides accuracy scores and confusion matrix for model evaluation.

---

## Results Summary
The trained Random Forest Classifier achieved the following results:
- **Accuracy**: 96.17%
- **Confusion Matrix**:
[[ 14 7]
[ 1 187]]

text
These results demonstrate the model's ability to accurately predict students' performance, with minimal false positives and false negatives.
