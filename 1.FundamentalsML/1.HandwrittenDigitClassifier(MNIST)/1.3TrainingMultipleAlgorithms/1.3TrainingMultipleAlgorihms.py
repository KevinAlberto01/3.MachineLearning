#1.IMPORT LIBRARIES
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier 

#2.LOAD DATASET
digits = load_digits()
x = digits.data #Images (8x8)
y = digits.target #Labels (0-9)

#3.SPLIT INTO TRAIN AND TEST SETS
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#4.NORMALIZE THE DATA
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)

#5.DEFINE THE MODELS 
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(kernel='linear'),
    "Multi-Layer Perception (MLP)": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
}

#6.TRAIN MODELS AND EVALUATE
results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    results[name] = accuracy
    print(f"{name} - Accuracy: {accuracy: .4f}")

#7.VISUALIZE THE RESULTS 
plt.figure(figsize=(8,5))   
plt.bar(results.keys(), results.values(), color = 'skyblue')
plt.ylabel('Accuracy')
plt.title('Compation of Algorithms (MNIST 8x8)')
plt.xticks(rotation = 0, fontsize=10)
plt.show()
