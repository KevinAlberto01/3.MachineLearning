import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cargar modelo entrenado
model = joblib.load('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/lightgbm_optuna_model.pkl')

# Cargar nombres de columnas usadas en el entrenamiento
expected_columns = joblib.load('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/feature_names.pkl')

# Crear un nuevo DataFrame con las mismas columnas
new_data = pd.DataFrame([{
    col: 0 for col in expected_columns  # valores de ejemplo
}])

new_data['Overall Qual'] = 7

# Cargar el scaler que se guardó
scaler = joblib.load('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/min_max_scaler.pkl')

# Predicción logarítmica
prediction_log = model.predict(new_data)

# Crear un DataFrame para la predicción y aplicar la desnormalización
predicted_df = pd.DataFrame(prediction_log, columns=['SalePrice_log'])


# Desnormalizar la predicción (usando el scaler)
# Aquí aseguramos que las dimensiones coincidan con el scaler (2 columnas)
predicted_df['Gr Liv Area_log'] = 0  # Añadimos una columna ficticia para mantener las dimensiones
predicted_rescaled = scaler.inverse_transform(predicted_df)

# Solo tomamos la columna de 'SalePrice_log' desnormalizada
prediction_rescaled = predicted_rescaled[:, 0]

# Invertir la transformación logarítmica
prediction = np.expm1(prediction_rescaled)

print("Predicción final en dólares:", prediction)

# Cargar los datos reales
file_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/AmesHousing.csv'
df = pd.read_csv(file_path)

#Seleccionar las columnas necesarias
x_test = df[expected_columns].copy()  # Estas son las features con las que entrenaste

#Predcir con tu modelo
prediction_log = model.predict(x_test)

#Desormalizar si usaste scaler 
predicted_df = pd.DataFrame(prediction_log, columns=['SalePrice_log'])
predicted_df['Gr Liv Area_log'] = 0  # para mantener la dimensión si tu scaler lo espera

prediction_rescaled = scaler.inverse_transform(predicted_df)
prediction_final = np.expm1(prediction_rescaled[:, 0])  # quitar log1p

#Comparar contra los valores reales
real_values = df['SalePrice']

df_comparison = pd.DataFrame({
    'Real': real_values,
    'Predicción': prediction_final
})

#Grafica para comprarar
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Real', y='Predicción', data=df_comparison, alpha=0.5)
plt.plot([df_comparison['Real'].min(), df_comparison['Real'].max()],
         [df_comparison['Real'].min(), df_comparison['Real'].max()],
         color='red', linestyle='--')
plt.title("Predicciones vs Valores Reales")
plt.xlabel("Valor Real")
plt.ylabel("Predicción del Modelo")
plt.grid(True)
plt.show()

# Calcular metricas de evaluacion(Cuantificar el rendimiento)
rmse = np.sqrt(mean_squared_error(real_values, prediction_final))
mae = mean_absolute_error(real_values, prediction_final)
r2 = r2_score(real_values, prediction_final)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R^2 Score: {r2:.4f}")

#Revisar que valores predice peor(para analisis de errores)
df_comparison['Error absoluto'] = np.abs(df_comparison['Real'] - df_comparison['Predicción'])
errores_mayores = df_comparison.sort_values('Error absoluto', ascending=False).head(10)
print(errores_mayores)

plt.figure(figsize=(10, 6))
sns.barplot(x='Error absoluto', y=errores_mayores.index, data=errores_mayores)
plt.title("Top 10 Peores Predicciones (Mayor Error Absoluto)")
plt.xlabel("Error Absoluto")
plt.ylabel("Índice de Fila")
plt.grid(True)
plt.show()

# Revisar qué valores predice mejor (menor error absoluto)
mejores_predicciones = df_comparison.sort_values('Error absoluto', ascending=True).head(10)
print("Predicciones con menor error absoluto:")
print(mejores_predicciones)

plt.figure(figsize=(10, 6))
sns.barplot(x='Error absoluto', y=mejores_predicciones.index, data=mejores_predicciones)
plt.title("Top 10 Mejores Predicciones (Menor Error Absoluto)")
plt.xlabel("Error Absoluto")
plt.ylabel("Índice de Fila")
plt.grid(True)
plt.show()
