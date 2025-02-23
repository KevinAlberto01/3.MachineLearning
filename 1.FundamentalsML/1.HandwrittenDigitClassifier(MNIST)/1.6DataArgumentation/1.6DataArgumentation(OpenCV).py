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

def apply_augmentation(img):
    # Rotación más ligera (-10 a 10 grados)
    angle = np.random.uniform(-10, 10)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    
    # Traslación más reducida (-1 a 1 píxel)
    tx, ty = np.random.randint(-1, 2, 2)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(rotated, M, (w, h))
    
    # Ruido gaussiano más leve (std=5 en lugar de 10)
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    noisy = np.clip(translated.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return noisy

# Cargar dataset
print("Loading Dataset MNIST 8x8...")
digits = load_digits()
x = digits.images  # Mantener la forma original
y = digits.target

# Aplicar Data Augmentation con OpenCV
augmented_images = []
augmented_labels = []
for img, label in zip(x, y):
    augmented = apply_augmentation(img)
    augmented_images.append(augmented)
    augmented_labels.append(label)

# Convertir a numpy array y normalizar
x_augmented = np.array(augmented_images).reshape(len(augmented_images), -1) / 255.0
x_original = x.reshape(len(x), -1) / 255.0  # Normalizar imágenes originales
y_augmented = np.array(augmented_labels)

# Combinar imágenes originales con aumentadas
x_combined = np.concatenate((x_original, x_augmented), axis=0)
y_combined = np.concatenate((y, y_augmented), axis=0)

# Dividir dataset en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x_combined, y_combined, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Definir modelos y ajustar hiperparámetros
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
    
    # Evaluar modelo
    y_pred = best_models[name].predict(x_test_scaled)
    accuracy = best_models[name].score(x_test_scaled, y_test)
    results[name] = accuracy
    conf_matrix[name] = confusion_matrix(y_test, y_pred)
    classification_reports[name] = classification_report(y_test, y_pred)
    print(f"Best Parameters for {name}: {grid_search.best_params_}")
    print(f"{name} - Accuracy after augmentation: {accuracy:.4f}")

# Visualización de comparación de modelos
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color='skyblue', edgecolor='none')
plt.ylabel('Accuracy')
plt.title('Comparison of Optimized Models with Data Augmentation (OpenCV)')
plt.ylim(0.0, 1.0)
plt.xticks(rotation=0)
plt.grid(False)  # Eliminar líneas punteadas o rectas
plt.show()

# Mostrar matriz de confusión para cada modelo
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

# Reporte de clasificación
for name, report in classification_reports.items():
    print(f"\n{name} - Classification Report:")
    print(report)
    print("-" * 50)

print("Optimization With Data Augmentation (OpenCV) Completed!")
