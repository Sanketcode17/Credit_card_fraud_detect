# Credit_card_fraud_detect
** Supervised and Unsupervised Models ** 
Credit Card Fraud Detection
Objective
The goal is to develop machine learning models (both supervised and unsupervised) to detect fraudulent transactions in credit card datasets, 
where fraud cases are significantly rarer than non-fraud cases (less than 0.1%). This project implements various preprocessing, modeling, evaluation, and visualization techniques to effectively identify these rare events.

# Code Details
# 1. Dataset Preparation
Objective: Load and preprocess the dataset (creditcard.csv) for analysis.
Steps:
Handle missing values.
Scale numerical features like Time and Amount using StandardScaler.
Address class imbalance using SMOTE.
Significance:
Ensures the dataset is clean and balanced for effective training.

# 2. Exploratory Data Analysis (EDA)
Objective: Analyze the data distribution, especially the imbalance between fraud and non-fraud cases.
Steps:
Visualize class distribution.
Summarize statistical characteristics of the dataset.
Significance:
Understand the dataset and detect any anomalies or patterns.

# 3. Supervised Learning Models
Objective: Train models to classify transactions as fraud or non-fraud.
Models:
A) Logistic Regression
B) XGBoost
C) Multi-layer Perceptron (MLP)
Evaluation:
Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
Hyperparameter tuning for model optimization.
# Significance:
Demonstrates the ability to classify transactions with standard machine learning techniques.

# 4. Unsupervised Learning Models
Objective: Use anomaly detection techniques to identify fraudulent transactions.
Models:
# A) Isolation Forest: Detects anomalies based on the decision function.
# B) Autoencoder: Learns a compressed representation and flags transactions with high reconstruction errors.
Significance:
Effectively identifies anomalies without labeled data.

# 5. Model Visualization
Objective: Visualize model performance using ROC-AUC and PR-AUC curves.
Steps:
Plot ROC curves to evaluate the trade-off between TPR and FPR.
Plot PR curves to understand the precision-recall balance.

** Significance: **
Helps to better interpret performance metrics, especially for imbalanced datasets.
General Observations
Data Imbalance: Fraudulent transactions are less than 0.1% of the dataset, making it critical to focus on metrics like Precision, Recall, and F1-Score rather than Accuracy.

# Supervised Models:
Logistic Regression and XGBoost performed well in distinguishing fraudulent transactions when combined with SMOTE.
XGBoost typically outperforms Logistic Regression due to its ability to handle non-linear relationships.
# Unsupervised Models:
Isolation Forest flagged anomalies effectively with a configurable contamination parameter.
Autoencoders showed promise in detecting rare events through reconstruction errors but required careful tuning of the threshold.

** Visualizations: **
PR-AUC was more informative than ROC-AUC due to the severe class imbalance.

# General Conclusion
Supervised vs. Unsupervised:

Supervised models are effective when labeled data is available, especially when combined with techniques like SMOTE to address class imbalance.
Unsupervised models are valuable for detecting unknown or novel fraud patterns without requiring labeled data.
Significance of Thresholds:

In both supervised and unsupervised methods, setting appropriate thresholds for anomaly scores or probabilities is crucial to minimize false positives.
Business Insights:

Detecting even a small percentage of fraud can significantly reduce financial losses.
Anomaly detection methods like Isolation Forest and Autoencoders can provide insights into atypical transaction patterns for further investigation.
Future Enhancements
Experiment with advanced anomaly detection methods like Variational Autoencoders or One-Class SVM.
Incorporate domain knowledge to fine-tune models for real-world applications.
Apply transfer learning techniques to leverage pre-trained models for fraud detection tasks.

This README provides a clear and structured guide to the project, its components, and its findings. 

# Logistic Regression Results:
Accuracy: 0.948542989290048
Precision: 0.9741353718272228
Recall: 0.9217345505617978
F1 Score: 0.9472107959875875
Confusion Matrix:
[[83058  2091]
 [ 6687 78753]]

# XGBoost Results:
Accuracy: 0.9996951737802555
Precision: 0.9993917559537735
Recall: 1.0
F1 Score: 0.9996957854585449
Confusion Matrix:
[[85097    52]
 [    0 85440]]

# MLP Results:
Accuracy: 0.9998358628047529
Precision: 0.9996723920063649
Recall: 1.0
F1 Score: 0.9998361691670372
Confusion Matrix:
[[85121    28]
 [    0 85440]]

# Isolation Forest Results:
Confusion Matrix:
[[281755   2560]
 [   203    289]]
Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.99      1.00    284315
           1       0.10      0.59      0.17       492

    accuracy                           0.99    284807
   macro avg       0.55      0.79      0.58    284807
weighted avg       1.00      0.99      0.99    284807


# Autoencoder Results:
Confusion Matrix:
[[281752   2563]
 [   206    286]]
Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.99      1.00    284315
           1       0.10      0.58      0.17       492

    accuracy                           0.99    284807
   macro avg       0.55      0.79      0.58    284807
weighted avg       1.00      0.99      0.99    284807
