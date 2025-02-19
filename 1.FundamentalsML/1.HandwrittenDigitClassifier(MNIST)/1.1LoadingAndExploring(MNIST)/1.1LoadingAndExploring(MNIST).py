import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

#1.Load the dataset (8x8)
digits = load_digits()
x = digits.data #Each image is 8x8 (64)
y = digits.target #Target (0-9)

#2.Exploring the dimensions of the dataset
print(f"Dimension of x: {x.shape}") #(1797, 64) "179 images with 64 pixels" 
print(f"Dimension of y: {y.shape}") #(1797) "1797 labels"

#3.Check the classes and the number of examples per class 
clases, count_classes = np.unique(y, return_counts=True)
print(f"Classes: {clases}")
print(f"Number of examples per class: {count_classes}")

#4.View the distribution of the classes
plt.figure(figsize=(8,5))
plt.bar(clases, count_classes, color = 'skyblue')
plt.xlabel('Digit')
plt.ylabel('Number of examples')
plt.title('Distribution of the classes')
plt.show()

#5.View some example images 
fig,axes = plt.subplots(2,5, figsize=(10, 5))
fig.suptitle("Examples of images")
for i, ax in enumerate(axes.ravel()):
    ax.imshow(x[i].reshape(8,8), cmap = 'gray')
    ax.set_title(f"label: {y[i]}")
    ax.axis('off')
plt.show()