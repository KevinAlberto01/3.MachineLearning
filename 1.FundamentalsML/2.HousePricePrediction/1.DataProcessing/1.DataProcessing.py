import pandas as pd 
import numpy as np 

# URL of the dataset 
url = "https://raw.githubusercontent.com/dsrscientist/DSData/master/housing.csv"

# Load the dataset
df = pd.read_csv(url)

# Show the first rows
print("First rows of the dataset")
print(df.head())

# Show general information
print("\nGeneral Info:")
print(df.info())

# 
print("\nDescriptive Statistics:")
print(df.describe())

