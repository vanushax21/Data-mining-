# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 18:03:10 2023

@author: shval
"""
#Prepared by:
# Ivan Shvalev (1673633)
# Pjotr Otten (1675530)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the dataset from the URL
data_url = "Diabetes.csv"
dataR = pd.read_csv(data_url)

# Feature Engineering: BMI categories
dataR['BMI_Category'] = pd.cut(dataR['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# Feature Engineering: Age groups
dataR['Age_Group'] = pd.cut(dataR['Age'], bins=[0, 20, 30, 40, 50, 60, 100], labels=['20s','30s', '40s','50s', '60s', '70+'])

# Feature Engineering: Blood Pressure
dataR['BloodPressure_Group'] = pd.cut(dataR['BloodPressure'], bins=[0, 60, 80, 90, 120], labels=['Low', 'Normal', 'Elevated', 'High'])

# Feature Engineering: Glucose
dataR['Glucose_Group'] = pd.cut(dataR['Glucose'], bins=[0, 80, 140, 200, 300], labels=['Low', 'Normal', 'High', 'Very High'])

# Define column names
columns = ['Glucose', 'BMI', 'Insulin', 'Pregnancies','SkinThickness' , 'BloodPressure', 'DiabetesPedigreeFunction', 'Age', 'BMI_Category', 'Age_Group','BloodPressure_Group','Glucose_Group','Outcome']

# Create a DataFrame with selected columns
df = pd.DataFrame(dataR, columns=columns)

# Encode categorical variables 'BMI_Category', 'Age_Group', 'BloodPressure_Group', and 'Glucose_Group'
df = pd.get_dummies(df, columns=['BMI_Category', 'Age_Group', 'BloodPressure_Group', 'Glucose_Group'], drop_first=True)

# Separate features (X) and target (y)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize features (mean=0, std=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a Random Forest Classification model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Additional Steps for Model Interpretation and Visualization
# 1. Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# 2. Print Classification Report
print("\nModel Performance:")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_rep)

# 3. Visualize Feature Importances (for understanding feature importance)
feature_importances = rf_classifier.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=X.columns, color='grey')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importances in Random Forest Classifier")
plt.show()

