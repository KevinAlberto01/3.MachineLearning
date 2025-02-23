import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Cargar dataset
digits = load_digits()
x = digits.images
y = digits.target

# Normalizar y redimensionar las imágenes
x = np.array(x).reshape(len(x), 8, 8, 1) / 16.0
y = keras.utils.to_categorical(y, 10)

# Dividir en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Construcción del modelo MLP
def build_mlp_model():
    model = keras.Sequential([
        layers.Flatten(input_shape=(8, 8, 1)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Crear y entrenar el modelo
mlp_model = build_mlp_model()
mlp_model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# Evaluación del modelo
y_pred = mlp_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Reporte de clasificación
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - MLP')
plt.show()
