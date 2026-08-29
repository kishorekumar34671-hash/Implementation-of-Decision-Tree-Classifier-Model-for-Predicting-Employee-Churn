# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the employee dataset containing employee details and churn status.

2.Preprocess the data and split it into training and testing sets.

3.Train a Decision Tree Classifier using the training data.

4.Predict employee churn on the test data and evaluate the model accuracy.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KISHORE KUMAR B
RegisterNumber: 212225240073



# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load dataset
df = pd.read_csv("Employee.csv")

print("First 5 records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# Step 3: Separate input features and target variable
# 'left' is the target column in your dataset

X = df.drop('left', axis=1)
y = df['left']


# Step 4: Convert categorical columns into numerical columns
# Departments and salary are categorical

X = pd.get_dummies(X, drop_first=True)


print("\nFeatures after encoding:")
print(X.head())


# Step 5: Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# Step 6: Create Decision Tree Classifier

model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42
)


# Step 7: Train the model

model.fit(X_train, y_train)


# Step 8: Predict employee churn

y_pred = model.predict(X_test)


# Step 9: Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(round(accuracy * 100, 2), "%")


# Step 10: Confusion Matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Step 11: Classification Report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Step 12: Compare actual and predicted values

result = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print("\nActual vs Predicted:")
print(result.head(10))

*/
```

## Output:
<img width="1920" height="1080" alt="Screenshot 2026-08-29 184508" src="https://github.com/user-attachments/assets/66dd8c7a-73a3-4ec5-8f56-3444eb7ecfcf" />
<img width="1920" height="1080" alt="Screenshot 2026-08-29 184532" src="https://github.com/user-attachments/assets/83c5f4b2-4134-4d21-bf52-d51e0d4b0848" />
<img width="1920" height="1080" alt="Screenshot 2026-08-29 184538" src="https://github.com/user-attachments/assets/19f170a5-843e-4b7f-adbc-50ae6bafc1e7" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
