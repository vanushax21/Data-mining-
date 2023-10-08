# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 23:31:10 2023

@author: shval
"""

#Prepared by:
# Ivan Shvalev (1673633)
# Pjotr Otten (1675530)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
data_url = "Diabetes.csv"  # Replace with the actual URL or local path
df = pd.read_csv(data_url)

# Feature Engineering: BMI categories
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# Feature Engineering: Age groups
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 30, 40, 50, 60, 100], labels=['20s','30s', '40s','50s', '60s', '70+'])

# Feature Engineering: Blood Pressure
df['BloodPressure_Group'] = pd.cut(df['BloodPressure'], bins=[0, 60, 80, 90, 120], labels=['Low', 'Normal', 'Elevated', 'High'])

# Feature Engineering: Glucose
df['Glucose_Group'] = pd.cut(df['Glucose'], bins=[0, 80, 140, 200, 300], labels=['Low', 'Normal', 'High', 'Very High'])

columns = ['Glucose', 'BMI', 'Insulin', 'Pregnancies', 'SkinThickness', 'BloodPressure', 'DiabetesPedigreeFunction', 'Age', 'BMI_Category', 'Age_Group','BloodPressure_Group','Glucose_Group','Outcome']
dfnew = pd.DataFrame(df, columns=columns)
dfnew = pd.get_dummies(dfnew, columns=['BMI_Category', 'Age_Group', 'BloodPressure_Group', 'Glucose_Group'], drop_first=True)

# Data preprocessing
X = dfnew.drop(['Outcome'], axis=1)
y = dfnew['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a K-nearest neighbors (KNN) model
k = 33  # Adjust the number of neighbors as needed
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

# Predictions on the test set
y_pred = knn_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(confusion)
print("\nClassification Report:")
print(classification_rep)

# Visualize the data 

# Plot the accuracy vs. number of neighbors 
# This can help you choose an optimal 'k' value.
k_values = range(1, 51)  # Adjust the range as needed
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.title("Accuracy vs. Number of Neighbors")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
