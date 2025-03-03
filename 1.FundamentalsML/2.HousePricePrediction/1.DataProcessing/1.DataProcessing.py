import pandas as pd 
import numpy as np 

#1.CARGE THE DATASET

# URL of the dataset 
file_path = "/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.DataProcessing/AmesHousing.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Show the first rows
print("First rows of the dataset")
print(df.head())

# Show general information
print("\nGeneral Info:")
print(df.info())

# Descriptive statistics 
print("\nDescriptive Statistics:")
print(df.describe())

#Check for null values 
print("\nValues missing in each columns")
print(df.isnull().sum()[df.isnull().sum()>0])

#Initial Shape 
print(f"\nInitial Shape: {df.shape}")

#2.INITIAL EXPLORING

