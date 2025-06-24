import joblib
import numpy as np

modelo_cargado = joblib.load("/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/gbm_random.pkl")
resultado = modelo_cargado.predict([[8]])  # por ejemplo con OverallQual = 8
print(resultado)

precio_real = 10 ** resultado
print(precio_real)