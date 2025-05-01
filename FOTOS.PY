import joblib

# Guarda el modelo con Optuna (mejor modelo)
joblib.dump(best_optuna, 'lightgbm_optuna_model.pkl')
# Cargar el modelo
best_model = joblib.load('lightgbm_optuna_model.pkl')


import pickle
# Guardar
with open('lightgbm_optuna_model.pkl', 'wb') as f:
    pickle.dump(best_optuna, f)
# Cargar
with open('lightgbm_optuna_model.pkl', 'rb') as f:
    best_model = pickle.load(f)
