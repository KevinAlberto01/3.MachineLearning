#IMPORT LIBRARIES 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import albumentations as A 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

#LOAD DATASET
print("Loading Dataset MINIST 8x8...")
digits = load_digits()
x = digits.images #keep the original shape 
y = digits.target

#DEFINE DATA ARGUMENTATION USING ALBUMENTATIONS
transform = A.Compose([
    A.ShiftScaleRotate(shift_limit = 0.01, scale_limit = 0.1, rotate_limit = 15, p = 0.5),
    A.GridDistortion(num_steps = 5, distort_limit = 0.3, p=0.5),
    A.ElasticTransform(alpha = 1, sigma = 50, p = 0.5) 
])

#APPLY ARGUMENTATION TO TRAINING DATA
argumented_images = []
argumented_labels = []
for img, label in zip(x, y):
    argumented = transform(image = img)['image']
    argumented_images.append(argumented)
    argumented_labels.append(label)

#CONVERT TO NUMPY ARRAY
x_argument = np.array(argumented_images).reshape(len(argumented_images), -1)
y_argument = np.array(argumented_labels)

#DIVIDE DATASET INTO TRAINING AND TESTING 
x_train, x_test, y_train, y_test = train_test_split(x_argument, y_argument, test_size=0.2, random_state = 42)

#NORMALIZE DATA 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#DEFINE MODELS WITH HYPERPARAMETERS TUNING
param_grid = {
    "Logistic Regression": {"C": [0.1, 1, 10], "max_iter": [1000, 5000]},
    "K-Nearest Neighbors (KNN)": {"n_neighbors": [3, 5, 7]}, "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "Multi-Layer Perceptron(MLP)": {"hidden_layer_sizes": [(50,), (100,)], "max_iter": [500, 1000]} 
}

best_models = {}
results = {}
conf_matrix = {}
classification_reports = {}

for name, params in param_grid.items():
    print(f"Optimizing {name}...")
    if name == "Logistic Regression":
        model = LogisticRegression()
    elif name == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier()
    elif name == "SVM":
        model = SVC()
    elif name == "Multi-Layer Perceptron(MLP)":
        model = MLPClassifier(random_state = 42)

    grid_search = GridSearchCV(model, params, cv = 3, scoring = 'accuracy')
    grid_search.fit(x_train, y_train)
    best_models[name] = grid_search.best_estimator_

    #EVALUATING MODEL 
    y_pred = best_models[name].predict(x_test)
    accuracy = best_models[name].score(x_test, y_test)
    results[name] = accuracy
    conf_matrix[name] = confusion_matrix(y_test, y_pred)
    classification_reports[name] = classification_report(y_test, y_pred)
    print(f"Best Parameters for {name}: {grid_search.best_params_}")
    print(f"{name} - Accuracy after argumentation: {accuracy: .4f}")

print ("Models in Results Dictionary:"), results.keys()
#VISUALIZATION OF ACCURACY COMPARISON 
plt.figure(figsize=(8,5 ))
plt.bar(results.keys(), results.values(), color = 'skyblue')
plt.ylabel('Accuracy')
plt.title('Comparison of Oprimizes Algorithms with Data Argumentation (Albumentations)')
plt.xticks(rotation = 0)
plt.ylim(0.75, 1.0)
plt.show()

#SHOW CONFUSION MATRIX FOR EACH MODEL
for name, matrix in conf_matrix.items():
    plt.figure(figsize=(6,5))
    sns.heatmap(matrix, annot = True, fmt = 'd', cmap = 'Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"Confusion Matrix - {name}")
    plt.show(block=True)

    print(f"Confusion Matrix - {name}:")
    print(matrix)
    print("-" * 50)

#CLASSIFICATION REPORT
for name, report in classification_reports.items():
    print(f"\n{name} - Classification Report:")
    print(report)
    print("-" * 50)

print("Optimization With Data Argumentation Completed!")