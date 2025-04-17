# Fraudulent E-Commerce Transactions Detection

This project focuses on detecting fraudulent transactions in e-commerce platforms using machine learning models. The goal is to develop a robust fraud detection system that minimizes false negatives while maintaining an acceptable balance with precision. This ensures the platform can prevent financial losses, mitigate risks, and enhance customer trust.

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

---

## Introduction

Fraudulent transactions are a significant challenge for e-commerce platforms. This project leverages machine learning techniques to classify transactions as fraudulent or legitimate based on various features. The project emphasizes minimizing false negatives to ensure fraudulent transactions are detected effectively.

---
## Problem Statement

The objective is to develop a machine learning model that:
- Accurately classifies transactions as fraudulent or legitimate.
- Minimizes false negatives to detect as many fraudulent transactions as possible.
- Maintains an acceptable balance with precision to reduce false positives.

---

## Repository Structure
### raw_data
Contains data files straight from Kaggle, with no preprocessing

### preprocessed_data
Contains data files after preprocessing step

### models
Contains our saved models

## How to Run

To run/test the app locally, follow these step-by-step instructions:

**Step 1:** Clone Git Repo  
Clone this Git repository containing the application source code to your local machine and run `cd ecommerce-fraud-detection`.

**Step 2:** Install necessary python packages  
Run the following command to install all the necessary python packages
```bash
pip install -r requirements.txt
```

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
- **Imbalanced-learn**: For handling imbalanced datasets

### Clustering and Anomaly Detection
- **KMeans**: For clustering.
- **GaussianMixture**: For probabilistic clustering.
- **IsolationForest**: For anomaly detection.

### Neural Networks
- **torch**: For building and training neural networks.

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



