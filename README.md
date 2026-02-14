# ML Assignment 2 - Classification Analysis

## 1. Problem Statement 
The goal of this project is to build a robust machine learning pipeline to classify [Insert Subject, e.g., breast cancer tumors] as [Class A vs Class B, e.g., Malignant or Benign]. This involves training multiple models, comparing their performance metrics, and deploying an interactive web application for real-time inference.

## 2. Dataset Description 
* **Source:** [e.g., Scikit-Learn / UCI Machine Learning Repository]
* **Name:** Breast Cancer Wisconsin (Diagnostic)
* **Features:** 30 numerical features computed from digitized images of fine needle aspirate (FNA) of a breast mass.
* **Instances:** 569 data points.
* **Target:** Binary classification (Malignant vs Benign).

## 3. Models Used & Comparison 

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.97 | 0.99 | 0.97 | 0.98 | 0.97 | 0.94 |
| Decision Tree | 0.94 | 0.93 | 0.94 | 0.94 | 0.94 | 0.88 |
| kNN | 0.96 | 0.98 | 0.96 | 0.95 | 0.96 | 0.92 |
| Naive Bayes | 0.93 | 0.98 | 0.93 | 0.91 | 0.92 | 0.86 |
| Random Forest | 0.98 | 0.99 | 0.98 | 0.98 | 0.98 | 0.95 |
| XGBoost | 0.97 | 0.99 | 0.97 | 0.97 | 0.97 | 0.94 |


## 4. Observations 

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performed excellently due to the linear separability of the features. |
| Decision Tree | Showed slight overfitting compared to ensemble methods. |
| KNN | Good performance, but sensitive to feature scaling (which was applied). |
| Naive Bayes | Slightly lower recall, assuming feature independence which may not hold fully. |
| Random Forest | The best performing model, robust to variance and high dimensionality. |
| XGBoost | Comparable to Random Forest, highly efficient and accurate. |

## 5. How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
