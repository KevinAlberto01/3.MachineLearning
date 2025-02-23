import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A 
import os
from tensorflow import keras 
from tensorflow.keras import layers 
from sklearn.datasets import load_digits 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Cargar datos
digits = load_digits()
x = digits.images
y = digits.target

# Data Augmentation
transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.5)
])

argumented_images = []
argumented_labels = []
for img, label in zip(x, y):
    augmented = transform(image=img)['image']
    argumented_images.append(augmented)
    argumented_labels.append(label)

x_argument = np.array(argumented_images).reshape(len(argumented_images), 8, 8, 1)
y_argument = keras.utils.to_categorical(np.array(argumented_labels), 10)

# Dividir datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x_argument, y_argument, test_size=0.2, random_state=42)

# Normalizar datos
x_train = x_train / 16.0
x_test = x_test / 16.0

# Definir el modelo KNN (red neuronal simple)
def build_knn_model():
    model = keras.Sequential([
        keras.Input(shape=(8, 8, 1)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# Entrenar el modelo
model = build_knn_model()
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Precisión en test: {test_acc:.4f}')

# Predicción
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Matriz de confusión
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - KNN')
plt.show()

# Reporte de clasificación
print("Classification Report - KNN:")
print(classification_report(y_test_classes, y_pred_classes))
