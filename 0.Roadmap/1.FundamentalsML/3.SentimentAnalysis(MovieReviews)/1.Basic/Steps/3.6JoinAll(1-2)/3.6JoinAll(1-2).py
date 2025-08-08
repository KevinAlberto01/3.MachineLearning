import joblib

# Cargar el pipeline guardado (puedes cargarlo fuera de la función para no cargarlo cada vez)
modelo_cargado = joblib.load('1.Basic/Steps/3.6JoinAll(1-2)/LogisticPipeline.pkl')

def predecir_sentimiento(texto):
    label_map = {0: 'Negative', 1: 'Positive'}
    prediccion = modelo_cargado.predict([texto])
    probabilidades = modelo_cargado.predict_proba([texto])
    
    prediccion_texto = label_map[prediccion[0]]
    prob_pos = probabilidades[0][1]  # Probabilidad de clase positiva
    
    return prediccion_texto, prob_pos

# Ejemplo de uso:
texto_nuevo = "This movie was fantastic! Loved every minute of it."
sentimiento, probabilidad = predecir_sentimiento(texto_nuevo)

print(f"Predicción: {sentimiento}")
print(f"Probabilidad de positivo: {probabilidad:.2%}")
