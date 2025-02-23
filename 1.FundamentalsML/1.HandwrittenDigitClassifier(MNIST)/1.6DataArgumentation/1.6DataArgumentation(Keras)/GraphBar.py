import matplotlib.pyplot as plt

# Resultados de precisión de cada modelo con Keras (sustituye estos valores con los reales)
results = {
    "Logistic Regression (Keras)": 0.85,
    "KNN (Keras)": 0.88,
    "SVM (Keras)": 0.87,
    "MLP (Keras)": 0.92
}

# Crear la gráfica de barras
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel('Accuracy')
plt.title('Comparison of Classification Models (Keras)')
plt.ylim(0, 1)

# Mostrar la gráfica
plt.show()
