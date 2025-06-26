#1.IMPORT LIBRARIES
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits 
from sklearn.preprocessing import StandardScaler

#2.LOAD DATA(MNIST)
digits = load_digits() #Dataset (8x8) of Sklearn 
x = digits.data #Images (each image is a vector of 64 elements (8x8))
y = digits.target #Labels (0-9)

#3.SHOW IMAGES BEFORE NORMALIZATION
fig, axes = plt.subplots(1, 5, figsize=(10, 4))
fig.suptitle("Examples of original images")
for i, ax in enumerate(axes):
    ax.imshow(x[i].reshape(8,8), cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
plt.show()

#4.NORMALIZATION WITH STANDARS SCALER
scaler  = StandardScaler()
x_scaled = scaler.fit_transform(x)

#5.SHOW THE FIRST ROWS OF THE NORMALIZED DATAS
print("Datas normalized (first 5 rows):")
print(x_scaled[:5])