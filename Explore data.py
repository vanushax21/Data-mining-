# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 12:20:39 2023

@author: shval
"""

#Prepared by:
# Ivan Shvalev (1673633)
# Pjotr Otten (1675530)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm

# Load the dataset
data_url = "Diabetes.csv"  # Replace with the actual URL or local path
df = pd.read_csv(data_url)

# Explore the dataset
print("Dataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Correlation matrix (numerical values only)
numeric = df.select_dtypes(include=['number'])
matrix = numeric.corr()

# Plot a heatmap to visualize correlations between features
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, annot=True, cmap="coolwarm", cbar=False)
plt.xlabel("Features")
plt.ylabel("Features")
plt.title("Correlation Matrix")
plt.show()

# Create a pairplot to visualize feature distributions with hue for 'Outcome'
sns.pairplot(df, hue='Outcome')
plt.show()


