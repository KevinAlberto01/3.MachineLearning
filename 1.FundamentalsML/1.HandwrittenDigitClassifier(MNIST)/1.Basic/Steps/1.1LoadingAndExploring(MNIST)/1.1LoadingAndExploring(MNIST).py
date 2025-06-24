#1.DECLARATE LIBRARIES
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

#2.LOAD THE DATASET(8x8)
digits = load_digits()
x = digits.data #Each image is 8x8 (64)
y = digits.target #Target (0-9)

#3.EXPLORING THE DIMENSIONS OF THE DATASET
print(f"Dimension of x: {x.shape}") #(1797, 64) "179 images with 64 pixels" 
print(f"Dimension of y: {y.shape}") #(1797) "1797 labels"

#4.CHECK THE CLASSES AND THE NUMBER OF EXAMPLES PER CLASS
clases, count_classes = np.unique(y, return_counts=True)
print(f"Classes: {clases}")
print(f"Number of examples per class: {count_classes}")

#5.VIEW THE DISTRIBUTION OF THE CLASSES
plt.figure(figsize=(8,5))
plt.bar(clases, count_classes, color = 'skyblue')
plt.xlabel('Digit')
plt.ylabel('Number of examples')
plt.title('Distribution of the classes')
plt.show()

#6.VIEW SOME EXAMPLE IMAGES  
fig,axes = plt.subplots(2,5, figsize=(10, 5))
fig.suptitle("Examples of images")
for i, ax in enumerate(axes.ravel()):
    ax.imshow(x[i].reshape(8,8), cmap = 'gray')
    ax.set_title(f"label: {y[i]}")
    ax.axis('off')
plt.show()