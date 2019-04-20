# Credit Card Fraud Detection

### Dataset : https://www.kaggle.com/mlg-ulb/creditcardfraud (Version 3)

### Objective: Correctly identifying fraudulent(1) and valid/legitimate(0) credit card transactions.

### Anomaly detection algorithms used:
1. Local Outlier Factor
2. Isolation Forest

LocalOutlierFactor gave an accuracy of 0.996 and IsolationForest gave an accuracy of 0.997. Both identified valid transactions with a precision of 1.00 but performed poorly in identifying fraudulent transactions having a precision of 0.03 by LocalOutlierFactor and 0.23 by IsolationForest.
