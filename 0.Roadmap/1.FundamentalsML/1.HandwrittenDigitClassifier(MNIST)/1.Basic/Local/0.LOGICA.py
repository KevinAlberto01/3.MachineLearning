#1.DECLARATE LIBRARIES
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier 
import seaborn as sns 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#1.1.LOAD THE DATASET(8x8)
digits = load_digits()
x = digits.data #Each image is 8x8 (64)
y = digits.target #Target (0-9)

#1.2.EXPLORING THE DIMENSIONS OF THE DATASET
print()
print(f"Dimension of x: {x.shape}") #(1797, 64) "179 images with 64 pixels" 
print(f"Dimension of y: {y.shape}") #(1797) "1797 labels"
print()

#1.3.CHECK THE CLASSES AND THE NUMBER OF EXAMPLES PER CLASS
print()
clases, count_classes = np.unique(y, return_counts=True)
print(f"Classes: {clases}")
print(f"Number of examples per class: {count_classes}")
print()

#1.4.VIEW THE DISTRIBUTION OF THE CLASSES
print()
plt.figure(figsize=(8,5))
plt.bar(clases, count_classes, color = 'skyblue')
plt.xlabel('Digit')
plt.ylabel('Number of examples')
plt.title('Distribution of the classes')
plt.show()
print()

#1.5.VIEW SOME EXAMPLE IMAGES  
fig,axes = plt.subplots(2,5, figsize=(10, 5))
fig.suptitle("Examples of images")
for i, ax in enumerate(axes.ravel()):
    ax.imshow(x[i].reshape(8,8), cmap = 'gray')
    ax.set_title(f"label: {y[i]}")
    ax.axis('off')
plt.show()


#2.1.NORMALIZATION WITH STANDARS SCALER
scaler  = StandardScaler()
x_scaled = scaler.fit_transform(x)

#2.2.SHOW THE FIRST ROWS OF THE NORMALIZED DATAS
print("Datas normalized (first 5 rows):")
print(x_scaled[:5])


#3.1.SPLIT INTO TRAIN AND TEST SETS
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#3.2.NORMALIZE THE DATA
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)

#3.3.DEFINE THE MODELS 
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(kernel='linear'),
    "Multi-Layer Perception (MLP)": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
}

#3.4.TRAIN MODELS AND EVALUATE
results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    results[name] = accuracy
    print(f"{name} - Accuracy: {accuracy: .4f}")

#3.5.VISUALIZE THE RESULTS 
plt.figure(figsize=(8,5))   
plt.bar(results.keys(), results.values(), color = 'skyblue')
plt.ylabel('Accuracy')
plt.title('Compation of Algorithms (MNIST 8x8)')
plt.xticks(rotation = 0, fontsize=10)
plt.show()


#4.1.TRAINING AND EVALUATE MODELS 
results = {}
conf_matrix = {}
classification_reports = {}
for name, model in models.items():
    
    print(f"Training {name}...")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    #6.1.ACCURACY
    accuracy = model.score(x_test, y_test)
    results[name] = accuracy
    
    #6.2 CONFUSION MATRIX
    conf_matrix[name] = confusion_matrix(y_test, y_pred)

    #6.3 CLASSIFICATION REPORT
    classification_reports[name] = classification_report(y_test, y_pred)
    print(f"{name} - Accuracy: {accuracy: .4f}")
print()

#4.2.SHOW CONFUSION MATRIX FOR EACH MODEL
for name, matrix in conf_matrix.items():
    plt.figure(figsize = (6,5))
    sns.heatmap(matrix, annot = True, fmt = 'd', cmap= 'Blues', 
                xticklabels=digits.target_names, yticklabels=digits.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Cofusion Matrix - {name}')
    plt.show() 

    #7.1 ADITIONAL INFORMATION
    print(f"Confusion Matrix for {name}:")
    print(matrix)
    print("\nMost misclassified digits:")

    #7.2 FIND THE MOST MISCLASSIFIED DIGITS
    errors = np.where(matrix != np.diag(matrix))
    for true_label, pred_label in zip(errors[0], errors[1]):
        print(f" - The model condused {true_label} with {pred_label} {matrix[true_label]} times.")
    print("-" * 40)
    
#4.3.CLASSIFICATION REPORT 
for name, report in classification_reports.items():
    print(f"\n{name} - Report of classification:")
    print(report)
    print("-" * 50)
    
print("Evaluation Completed!")




#5.1.DEFINE MODELS WITH HYPERPARAMETERS TUNING 
param_grid = {
    "Logistic Regression": {"C": [0.1, 1, 10], "max_iter": [1000, 5000]},
    "K-Nearest Neighbors (KNN)": {"n_neighbors": [3, 5, 7]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "Multi-Layer Perceptron (MLP)": {"hidden_layer_sizes": [(50,),(100,)],"max_iter":[500,1000]}
}
best_models = {}
results = {}
conf_matrix = {}
classification_reports = {}

#5.2.OPTIMIZE MODELS
for name, params in param_grid.items():
    print(f"Optimizing {name}...")
    if name == "Logistic Regression":
        model = LogisticRegression()
    elif name == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier()
    elif name == "SVM":
        model = SVC()
    elif name == "Multi-Layer Perceptron (MLP)":
        model = MLPClassifier(random_state = 42)

    grid_search = GridSearchCV(model, params, cv=3, scoring = 'accuracy')
    grid_search.fit(x_train, y_train)
    best_models[name] = grid_search.best_estimator_

    #EVALUATE BEST MODEL
    y_pred = best_models[name].predict(x_test)
    accuracy = best_models[name].score(x_test, y_test)
    results[name] = accuracy 
    conf_matrix[name] = confusion_matrix(y_test, y_pred)
    classification_reports[name] = classification_report(y_test, y_pred)
    print(f"Best Parameters for {name}: {grid_search.best_params_}")
    print(f"{name} - Accuracy: {accuracy: .4f}")

#5.3.VISUALIZATION COMPARATION OF ACCURACY 
plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values(), color = 'skyblue')
plt.ylabel('Accuracy')
plt.title('Comparation of Optimized Algorithms (MNIST 8X8)')
plt.xticks(rotation = 0)
plt.ylim(0.95, 1.0)
plt.show()

#5.4 SHOW CONFUSION MATRIX FOR EACH MODEL
for name, matrix in conf_matrix.items():
    plt.figure(figsize = (6,5))
    sns.heatmap(matrix, annot = True, fmt= 'd', cmap = 'Blues', 
                xticklabels= digits.target_names, yticklabels= digits.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {name}:')
    plt.show()
    print(f"Confusion Matrix for {name}:")
    print(matrix)
    print("\nMost Misclassified Digits:")
    #7.2 Error Analysis
    errors = np.where(matrix != np.diag(matrix))
    for true_label, pred_label in zip(errors[0], errors[1]):
        print (f" - The model confused {true_label} with {pred_label} {matrix[true_label, pred_label]} times.")
    print("-" * 50)

#5.5.CLASSIFICATION REPORT 
for name, report in classification_reports.items():
    print(f"\n{name} - Classification Report:")
    print(report)
    print("-" * 50)
print("Optimization Completed!")