#1.IMPORT LIBRARIES
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

#2.LOAD DATASET 
print("Loading dataset MNIST 8x8...")
digits = load_digits()
x = digits.data
y = digits.target

#3.DIVIDE DATASET INTO TRAINING AND TESTING
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#4.NORMALIZE DATA 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#5.DEFINE MODELS WITH HYPERPARAMETERS TUNING 
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

#6.OPTIMIZE MODELS
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

#7.VISUALIZATION COMPARATION OF ACCURACY 
plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values(), color = 'skyblue')
plt.ylabel('Accuracy')
plt.title('Comparation of Optimized Algorithms (MNIST 8X8)')
plt.xticks(rotation = 0)
plt.ylim(0.95, 1.0)
plt.show()

#7.1 SHOW CONFUSION MATRIX FOR EACH MODEL
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

#8.CLASSIFICATION REPORT 
for name, report in classification_reports.items():
    print(f"\n{name} - Classification Report:")
    print(report)
    print("-" * 50)
print("Optimization Completed!")