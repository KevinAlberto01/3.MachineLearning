#1.IMPORT LIBRARIES
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#1.LOAD DATASET
print("Loading dataset MNIST 8x8...")
digits = load_digits()
x = digits.data #Images (8x8)
y = digits.target #Labels (0-9)

#2.DIVIDE DATASET INTO TRAINING AND TESTING 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#3.NORMALIZE DATA
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#4.DEFINE MODELS 
models = { 
    "Logistic Regression": LogisticRegression(max_iter = 5000),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors = 3),
    "SVM": SVC(kernel = 'linear'),
    "Multi-Layer Perceptron (MLP)": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state = 42)

}

#5.TRAINING AND EVALUATE MODELS 
results = {}
conf_matrix = {}
classification_reports = {}
for name, model in models.items():
    
    print(f"Training {name}...")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    #5.1.ACCURACY
    accuracy = model.score(x_test, y_test)
    results[name] = accuracy
    
    #5.2 CONFUSION MATRIX
    conf_matrix[name] = confusion_matrix(y_test, y_pred)

    #5.3 CLASSIFICATION REPORT
    classification_reports[name] = classification_report(y_test, y_pred)
    print(f"{name} - Accuracy: {accuracy: .4f}")

print()

#6.VISUALIZATION COMPARATION OF ACURRACY
plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values(), color = 'skyblue')
plt.ylabel('Acurracy')
plt.title('Comparation of algorithms (MNIST 8x8)')
plt.xticks(rotation = 0) #Labels in Straight Line
plt.ylim(0.95, 1.0) #Adjust scale to have a better visualization
plt.show()
print()

#7.SHOW CONFUSION MATRIX FOR EACH MODEL
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
    
#8.CLASSIFICATION REPORT 
for name, report in classification_reports.items():
    print(f"\n{name} - Report of classification:")
    print(report)
    print("-" * 50)
    
print("Evaluation Completed!")
