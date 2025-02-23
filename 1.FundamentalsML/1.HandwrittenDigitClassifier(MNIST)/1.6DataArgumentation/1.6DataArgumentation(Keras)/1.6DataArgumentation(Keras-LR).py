import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import os
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Configurar TensorFlow para GPU (opcional)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Cargar dataset
digits = load_digits()
x = digits.images  # Mantener la forma original
y = digits.target

# Data Augmentation con Albumentations
transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.5)
])

augmented_images = []
augmented_labels = []
for img, label in zip(x, y):
    augmented = transform(image=img)['image']
    augmented_images.append(augmented)
    augmented_labels.append(label)

# Convertir a numpy array
x_augmented = np.array(augmented_images).reshape(len(augmented_images), 8, 8, 1)
y_augmented = keras.utils.to_categorical(np.array(augmented_labels), 10)

# Dividir dataset en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x_augmented, y_augmented, test_size=0.2, random_state=42)

# Normalizar datos
x_train = x_train / 16.0
x_test = x_test / 16.0

# Definir modelo Logistic Regression en Keras
def build_logistic_regression(input_shape):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# Crear y entrenar modelo
model = build_logistic_regression((8, 8, 1))
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# Evaluar modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Predicciones
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Matriz de confusión
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Reporte de clasificación
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes))

# Graficar matriz de confusión
import seaborn as sns
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

