import matplotlib.pyplot as plt
import numpy as np

# Datos de accuracy
models = ["Logistic Regression", "K-Nearest Neighbors (KNN)", "SVM", "Multi-Layer Perceptron (MLP)"]
accuracy = [0.68, 0.86, 0.91, 0.97]

# Crear la gráfica
plt.figure(figsize=(8,5))
plt.bar(models, accuracy, color='skyblue')
plt.xlabel("Modelos")
plt.ylabel("Accuracy")
plt.title("Comparación de Accuracy entre Modelos")
plt.ylim(0.6, 1.0)  # Ajuste del rango para incluir todos los valores

# Mostrar la gráfica
plt.show()
