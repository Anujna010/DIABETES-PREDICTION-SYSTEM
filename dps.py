import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('diabetes.csv')

# DESCRIPTIVE STATICSTICS 
# Preview data
print("Preview data")
print(dataset.head())

# Dataset dimensions
print("Dataset dimensions")
print(dataset.shape)

# Features data - type
print("Features data - type")
print(dataset.info())

# Statistical Summary 
print("Statistical Summary ")
print(dataset.describe().T)

# Count of all null values 
print("Count of all null values ")
dataset.isnull().sum()

# DATA PREPROCESSING 

dataset_new = dataset

# Replace zero with NaN

dataset_new[["Pregnancies" , "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI" ,
              "DiabetesPedigreeFunction", "Age"]] = dataset_new[["Pregnancies","Glucose", "BloodPressure",
             "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]].replace(0, np.NaN) 

# Replacing NaN with mean values
dataset_new["Pregnancies"].fillna(dataset_new["Pregnancies"].mean(), inplace = True)
dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)
dataset_new["DiabetesPedigreeFunction"].fillna(dataset_new["DiabetesPedigreeFunction"].mean(), inplace = True)
dataset_new["Age"].fillna(dataset_new["Age"].mean(), inplace = True)

# Statistical summary
print("Statistical summary")
print(dataset_new.describe().T)