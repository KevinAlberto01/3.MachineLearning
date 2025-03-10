#1.LIBRARY IMPORTATION
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

#2.DATA AUGMENTATION FUNCTION WITH OPEN CV
def apply_augmentation(img):
    # Lighter rotation (-10 to 10 degrees)
    angle = np.random.uniform(-10, 10)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    
    # Reduced translation (-1 to 1 pixel)
    tx, ty = np.random.randint(-1, 2, 2)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(rotated, M, (w, h))
    
    # Softer Gaussian noise (std=5 instead of 10)
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    noisy = np.clip(translated.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return noisy

#3.LOAD DATASET
print("Loading Dataset MNIST 8x8...")
digits = load_digits()
x = digits.images  # Mantener la forma original
y = digits.target

#4.APPLY DATA AUGMENTATION WITH OPEN CV 
augmented_images = []
augmented_labels = []
for img, label in zip(x, y):
    augmented = apply_augmentation(img)
    augmented_images.append(augmented)
    augmented_labels.append(label)

#5.CONVERT TO NUMPY ARRAY AND NORMALIZE 
x_augmented = np.array(augmented_images).reshape(len(augmented_images), -1) / 255.0
x_original = x.reshape(len(x), -1) / 255.0  # Normalizar imágenes originales
y_augmented = np.array(augmented_labels)

#6.COMBINE ORIGINAL IMAGES WITH AUGMENTED ONES
x_combined = np.concatenate((x_original, x_augmented), axis=0)
y_combined = np.concatenate((y, y_augmented), axis=0)

#7.SPLIT DATASET INTO TRAINING AND TEST
x_train, x_test, y_train, y_test = train_test_split(x_combined, y_combined, test_size=0.2, random_state=42)

#8.SCALE THE DATA
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#9.DEFINE MODELS AND ADJUST HYPERPARAMETERS
param_grid = {
    "Logistic Regression": {"C": [0.1, 1, 10], "max_iter": [1000, 5000]},
    "K-Nearest Neighbors (KNN)": {"n_neighbors": [3, 5, 7]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "Multi-Layer Perceptron (MLP)": {"hidden_layer_sizes": [(30,), (50,), (100,)], "max_iter": [2000, 5000]}
}

best_models = {}
results = {}
conf_matrix = {}
classification_reports = {}

#10.TRAIN & EVALUATE MODELS
for name, params in param_grid.items():
    print(f"Optimizing {name}...")
    if name == "Logistic Regression":
        model = LogisticRegression()
    elif name == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier()
    elif name == "SVM":
        model = SVC()
    elif name == "Multi-Layer Perceptron (MLP)":
        model = MLPClassifier(solver='adam', hidden_layer_sizes=(50,), max_iter=5000, random_state=42)
    
    grid_search = GridSearchCV(model, params, cv=3, scoring='accuracy')
    grid_search.fit(x_train_scaled, y_train)
    best_models[name] = grid_search.best_estimator_
    
    #10.1 Evaluate the model
    y_pred = best_models[name].predict(x_test_scaled)
    accuracy = best_models[name].score(x_test_scaled, y_test)
    results[name] = accuracy
    conf_matrix[name] = confusion_matrix(y_test, y_pred)
    classification_reports[name] = classification_report(y_test, y_pred)
    print(f"Best Parameters for {name}: {grid_search.best_params_}")
    print(f"{name} - Accuracy after augmentation: {accuracy:.4f}")

#11.MODEL COMPARISON VISUALIZATION
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color='skyblue', edgecolor='none')
plt.ylabel('Accuracy')
plt.title('Comparison of Optimized Models with Data Augmentation (OpenCV)')
plt.ylim(0.0, 1.0)
plt.xticks(rotation=0)
plt.grid(False)  # Eliminar líneas punteadas o rectas
plt.show()

#12. SHOW CONFUSION MATRIX FOR EACH MODELS
for name, matrix in conf_matrix.items():
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names, cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"Confusion Matrix - {name}")
    plt.grid(False)  # Eliminar líneas rectas
    plt.show()
    print(f"Confusion Matrix - {name}:")
    print(matrix)
    print("-" * 50)

#13.CLASSIFICATION REPORT
for name, report in classification_reports.items():
    print(f"\n{name} - Classification Report:")
    print(report)
    print("-" * 50)

print("Optimization With Data Augmentation (OpenCV) Completed!")
