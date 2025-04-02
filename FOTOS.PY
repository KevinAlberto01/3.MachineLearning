from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Crear el modelo Ridge
ridge_model = Ridge(alpha=1.0)  # alpha es el parámetro de regularización
# Entrenar el modelo
ridge_model.fit(x_train, y_train)
# Realizar predicciones
y_pred_ridge = ridge_model.predict(x_test)
# Calcular el MSE
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f'MSE for Ridge Regression: {mse_ridge}')

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Crear el modelo Lasso
lasso_model = Lasso(alpha=0.1)  # alpha es el parámetro de regularización
# Entrenar el modelo
lasso_model.fit(x_train, y_train)
# Realizar predicciones
y_pred_lasso = lasso_model.predict(x_test)
# Calcular el MSE
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f'MSE for Lasso Regression: {mse_lasso}')
