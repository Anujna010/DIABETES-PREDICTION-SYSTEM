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

# Feature scaling using Min max scaler 

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_new)

dataset_scaled = pd.DataFrame(dataset_scaled)

# Selecting features - [Glucose, Insulin, BMI, Age]
X = dataset_scaled.iloc[:, [1, 2 , 4 , 5 , 7]].values
Y = dataset_scaled.iloc[:, 8].values

# Splitting X and Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'] )

# Checking dimensions

print("Checking dimensions\n")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)