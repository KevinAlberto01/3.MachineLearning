#1.IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#2.LOAD DATASET
digits = load_digits()
x = digits.images.reshape(len(digits.images), 8, 8, 1)  # Redimensionar para Keras
y = keras.utils.to_categorical(digits.target, 10)  # One-hot encoding

#3.DIVIDE DATASET
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#4.NORMALIZATION OF DATA
x_train = x_train / 16.0
x_test = x_test / 16.0

#5.VERIFICATION OF THE DISTRIBUTION OF CLASSES
unique_train, counts_train = np.unique(np.argmax(y_train, axis=1), return_counts=True)
print("Clases en entrenamiento:", dict(zip(unique_train, counts_train)))

unique_test, counts_test = np.unique(np.argmax(y_test, axis=1), return_counts=True)
print("Clases en prueba:", dict(zip(unique_test, counts_test)))

#6.CONSTRUCTION OF THE MODEL (SVM WITH KERAS)
model = keras.Sequential([
    layers.Flatten(input_shape=(8, 8, 1)),
    layers.Dense(10, activation='softmax')  # Softmax para mejorar la clasificación
])

#7.COMPILATION OF MODEL
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#8.TRAINNING OF MODEL
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

#9.EVALUATION OF MODEL 
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

#10.MATRIX OF CONFUSION
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - SVM with Keras')
plt.show()

#11.REPORT OF CLASSIFICATION
print("Classification Report - SVM with Keras")
print(classification_report(y_test_classes, y_pred_classes, zero_division=1))
