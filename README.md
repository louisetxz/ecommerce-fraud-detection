# Fraudulent E-Commerce Transactions Detection

This project focuses on detecting fraudulent transactions in e-commerce platforms using machine learning models. The goal is to develop a robust fraud detection system that minimizes false negatives while maintaining an acceptable balance with precision. This ensures the platform can prevent financial losses, mitigate risks, and enhance customer trust.

## Table of Contents
- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Problem Statement](#problem-statement)
- [Features and Engineering](#features-and-engineering)
- [Models and Techniques](#models-and-techniques)
- [Results](#results)
- [Explainable AI (XAI)](#explainable-ai-xai)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

---

## Introduction

Fraudulent transactions are a significant challenge for e-commerce platforms. This project leverages machine learning techniques to classify transactions as fraudulent or legitimate based on various features. The project emphasizes minimizing false negatives to ensure fraudulent transactions are detected effectively.

---

## Dataset Overview

The dataset contains transactional data from an e-commerce platform, including engineered features to support fraud detection. 

- **Number of Transactions in Version 1**: 1,472,952
- **Number of Transactions in Version 2**: 23,634
- **Features**: 16
- **Fraudulent Transactions**: Approximately 5%

---

## Problem Statement

The objective is to develop a machine learning model that:
- Accurately classifies transactions as fraudulent or legitimate.
- Minimizes false negatives to detect as many fraudulent transactions as possible.
- Maintains an acceptable balance with precision to reduce false positives.

---

## Features and Engineering

Key features engineered for this project include:
1. **Single Item Transaction**: Ratio of transaction amount to quantity.
2. **Time of Day**: Categorized into Early Morning, Morning, Afternoon, and Night.
3. **Same Address**: Binary feature indicating whether the shipping and billing addresses match.
4. **Private IP Address**: Binary feature indicating if the IP address is private.
5. **Age Binning**: Categorized customer age into discrete bins.

---

## Models and Techniques

### Machine Learning Models
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting
- Neural Networks

### Techniques
- **Clustering**: KMeans, Gaussian Mixture Models (GMM), Isolation Forest.
- **Oversampling**: Random oversampling to handle class imbalance.
- **Feature Selection**: SelectKBest for dimensionality reduction.
- **Cross-Validation**: Stratified K-Fold with nested feature engineering and clustering.
- **Hyperparameter Tuning**: RandomizedSearchCV for optimal model parameters.

---

## Results

- **Best Model**: XGBoost
  - **F2-Score**: 0.7029
  - **Recall**: 0.7209
  - **Precision**: Balanced with recall.

- **Neural Network**:
  - **F2-Score**: 0.6802
  - **ROC-AUC**: 0.7870

- **Ensemble Model**:
  - Combined Random Forest and XGBoost for enhanced performance.

---

## Explainable AI (XAI)

LIME (Local Interpretable Model-agnostic Explanations) was used to explain model predictions. Key insights:
1. High transaction amounts are strong indicators of fraud.
2. Newer accounts are more likely to be fraudulent.
3. Mismatched shipping and billing addresses often indicate fraud.
4. Transactions during unusual hours (e.g., late night) are more suspicious.

---

## How to Run

To run/test the app locally, follow these step-by-step instructions:

**Step 1:** Clone Git Repo
Clone this Git repository containing the application source code to your local machine.

**Step 2:** Create a Virtual Environment
Set up a virtual environment for the application to ensure isolated dependencies.

**Step 3:** Start Docker Compose
Open a terminal and navigate to the root directory of the cloned repository.
Run the following command to start Docker Compose:
```bash
docker-compose up
```
**Step 4:** Wait for Docker Containers
Wait for Docker to create the necessary containers for the application.

---

## Dependencies

This project relies on the following Python libraries and frameworks:

### Core Libraries
- **Python 3.8+**: The programming language used for this project.
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.
- **Seaborn**: For advanced data visualization.

### Machine Learning Libraries
- **Scikit-learn**: For machine learning models, preprocessing, and evaluation metrics.
- **XGBoost**: For gradient boosting models.
- **LightGBM**: For efficient gradient boosting models.
- **Imbalanced-learn**: For handling imbalanced datasets (e.g., SMOTE, RandomOverSampler).

### Clustering and Anomaly Detection
- **KMeans**: For clustering.
- **GaussianMixture**: For probabilistic clustering.
- **IsolationForest**: For anomaly detection.

### Neural Networks
- **PyTorch**: For building and training neural networks.

### Explainable AI
- **LIME**: For generating interpretable explanations of model predictions.

### Utilities
- **Joblib**: For saving and loading trained models.
- **Kneed**: For determining the optimal number of clusters using the elbow method.


---

## Acknowledgments

We would like to acknowledge the following resources and contributors:

- Dataset: The "Fraudulent E-Commerce Transactions" dataset used in this project was provided for research purposes. It contains transaction data with features engineered for fraud detection.
- Scikit-learn Documentation: For providing comprehensive guides and examples for machine learning models and preprocessing techniques.
- XGBoost and LightGBM Teams: For developing powerful gradient boosting frameworks that significantly improved model performance.
- PyTorch Community: For creating a flexible and efficient deep learning framework.
- LIME Developers: For enabling interpretable machine learning through the LIME library.
- Open-Source Community: For maintaining and contributing to the libraries used in this project.

This project was developed as part of the coursework for the DSA4263 module at the National University of Singapore (NUS). Special thanks to the instructors and peers for their guidance and support. 



