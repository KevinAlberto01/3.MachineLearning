#0.IMPORT LIBRARIES
import pandas as pd

#1.1 LOAD DATASET
file_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/AmesHousing.csv'
df = pd.read_csv(file_path)

raws, columns = df.shape
print(f"Numers of rows: {raws}")
print(f"Numers of columns: {columns}")

print(df.head())