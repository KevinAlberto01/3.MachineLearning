#------------------------------------------------ 0.IMPORT LIBRARIES -----------------------------------------------# \
import os
import pandas as pd
import pickle
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import optuna
import joblib
import altair as alt
import streamlit.components.v1 as components
import lightgbm as lgb
from scipy.stats import gaussian_kde
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler ##
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from lightgbm import early_stopping, log_evaluation 
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from streamlit_option_menu import option_menu
#--------------------------------------------------------------------------------------------------------------------# 

#-------------------------------------------------- 1.DEVELOPMENT ---------------------------------------------------#
################################################## 1.1.LOAD DATASET ##################################################
file_path = 'data/AmesHousing.csv'
df = pd.read_csv(file_path)
######################################################################################################################

############################################ 1.2 EXPLORATORY DATA ANALYSIS ###########################################
selected_features = ['SalePrice', 'Overall Qual', 'Gr Liv Area', 'Garage Cars', 
                     'Garage Area', 'Total Bsmt SF', '1st Flr SF', 'Full Bath', 'Year Built']

df_encoded = pd.get_dummies(df, drop_first=True)
df_corr = df_encoded.corr()

df['SalePrice_log'] = np.log1p(df['SalePrice'])
df['Gr Liv Area_log'] = np.log1p(df['Gr Liv Area'])

scaler = MinMaxScaler()
df[['SalePrice_log', 'Gr Liv Area_log']] = scaler.fit_transform(df[['SalePrice_log', 'Gr Liv Area_log']])
######################################################################################################################

############################################ 1.3 TRAINING MULTIPLE MODELS ###########################################
y = df['SalePrice_log']
x2 = df[['Overall Qual']]
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, random_state = 42)
gbm = lgb.LGBMRegressor(objective = 'regression', random_state = 42, verbosity = -1)
gbm.fit(x2_train, y_train)
y_pred_basic = gbm.predict(x2_test)

def evalute_model(model, x_test, y_test, y_pred, model_name, feature):
    mae2 = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<p style='margin: 0; font-size: 18px; text-align: center;'><strong>Mean Absolute Error (MAE)</strong></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin: 0; font-size: 18px; text-align: center;'>{mae2:.4f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin: 0; font-size: 18px; text-align: center;'><strong>Root Mean Squared Error (RMSE)</strong></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin: 0; font-size: 18px; text-align: center;'>{rmse:.4f}</p>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<p style='margin: 0; font-size: 18px; text-align: center;'><strong>Mean Squared Error (MSE)</strong></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin: 0; font-size: 18px; text-align: center;'>{mse:.4f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin: 0; font-size: 18px; text-align: center;'><strong>R-Squared (R2)</strong></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin: 0; font-size: 18px; text-align: center;'>{r2:.4f}</p>", unsafe_allow_html=True)

def objetive(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30)
    }

    model = lgb.LGBMRegressor(**param)
    model.fit(x2_train, y_train)
    preds = model.predict(x2_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return rmse
######################################################################################################################

############################################# 1.4 LOAD TRAINING MODEL ###########################################
@st.cache_resource

def load_model():
    # Cargar modelo entrenado
    model = joblib.load('models/lightgbm_optuna_model.pkl')

    # Cargar nombres de columnas usadas en el entrenamiento
    expected_columns = joblib.load('models/feature_names.pkl')

    # Cargar el scaler que se guardó
    scaler = joblib.load('models/min_max_scaler.pkl')
    return model, expected_columns, scaler

#--------------------------------------------------------------------------------------------------------------------# 

model, expected_columns, scaler = load_model()  # Aquí sí puede estar cacheado por dentro
# Crear datos con columnas esperadas
new_data = pd.DataFrame([{col: 0 for col in expected_columns}])
prediction_before = 0

#Seleccionar las columnas necesarias
x_test = df[expected_columns].copy()  # Estas son las features con las que entrenaste


#-------------------------------------------------- 2.PAGE CONFIGURATION ---------------------------------------------------#
#################################################### 2.1 Principal page #####################################################

# Configurar página en modo ancho
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")
# Titulo de la pagina
st.markdown("<h1 style = 'text-align: center;'>🏠 House Price Prediction 🏠</h1>", unsafe_allow_html=True)
#Titulos de seleccion 

# Crear columnas vacías a los lados para centrar el radio
col1, col2, col3 = st.columns([3, 1.15, 3])  # Ajusta proporciones si quieres

with col2:
    section = st.radio(
        label="",
        options=["Results", "Development"],
        horizontal=True,
        label_visibility="collapsed"
    )

# Estilos personalizados (centrar título y tabla más ancha)
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .dataframe-container {
            width: 100% !important;
        }
    </style>
""", unsafe_allow_html=True)
##############################################################################################################################

########################################################## 2.2 Graphs ########################################################
def make_donut(input_response, input_text, input_color):
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    elif input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    elif input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    elif input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']
    
    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100 - input_response, input_response]
    })
    
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })

    plot = alt.Chart(source).mark_arc(innerRadius=50, cornerRadius=75).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(domain=[input_text, ''], range=chart_color),
                        legend=None)
    ).properties(width=300, height=250)

    text = plot.mark_text(
        align='center',
        color=chart_color[0],
        font="Lato",
        fontSize=32,
        fontWeight=700,
        fontStyle="italic"
    ).encode(text=alt.value(f'{input_response:.1f}%'))

    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=50, cornerRadius=75).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(domain=[input_text, ''], range=chart_color),
                        legend=None)
    ).properties(width=300, height=250)

    return plot_bg + plot + text
##############################################################################################################################
#----------------------------------------------------------------------------------------------------------------------------#



#----------------------------------------------------- 3.PROYECT PAGE -------------------------------------------------------#

if section == "Results":

    col1, col2, col3 = st.columns([0.9,1.2,0.9])
    with col1:

        ######################################################## 3.1 Seleccion del modelo ######################################################
        # CSS para eliminar márgenes y espacio extra entre los elementos
        st.markdown("<h3 style='font-size: 24px; margin: 0; padding: 0; text-align: center;'>Selección de Modelo</h3>", unsafe_allow_html=True)
        options = [
            "📊 Basic LGBM Regressor 📊", "📈 LGBM Regressor with Optuna 📈", "📉 LGBM Regressor with Early Stopping📉"
        ]
        selected_option = st.selectbox("", options, label_visibility="collapsed")

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++ 3.1.1 Basic LGM Regressor ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if selected_option == "📊 Basic LGBM Regressor 📊":
            col32, col20, col21 = st.columns([1, 2, 1])
            with col20:
                sections = st.radio(
                    label="",
                    options=["Results", "Real Resutls"],
                    horizontal=True,
                    label_visibility="collapsed"
                )

            if sections == "Results":

                mse = mean_squared_error(y_test, y_pred_basic)
                evalute_model(gbm, x2_test, y_test, y_pred_basic, "LightGBM Regressor", "Overall Qual" )

            elif sections == "Real Resutls":
                #exponencional
                y_pred_early_original = np.expm1(y_pred_basic)  # Exponencial inversa de la predicción
                y_test_early_original = np.expm1(y_test)
                evalute_model(gbm, x2_test, y_test_early_original, y_pred_early_original, "LightGBM Regressor without exponetional", "Overall Qual")
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++ 3.1.2 LGBM Regressor with Optuna ++++++++++++++++++++++++++++++++++++++++++++++++++
        elif selected_option == "📈 LGBM Regressor with Optuna 📈":
            study = optuna.create_study(direction = 'minimize')
            study.optimize(objetive, n_trials = 50)
                        
            #Evaluar el modelo
            best_params_optuna = study.best_params
            best_optuna = lgb.LGBMRegressor(n_estimators = best_params_optuna['n_estimators'], max_depth = best_params_optuna['max_depth'], learning_rate = best_params_optuna['learning_rate'], min_child_samples = best_params_optuna['min_child_samples'], random_state = 42)
            best_optuna.fit(x2_train, y_train)

            y_pred_optuna = best_optuna.predict(x2_test)

            cols = st.columns(len(study.best_params))

            for col, (key, value) in zip(cols, study.best_params.items()):
                col.metric(label=key, value=round(value, 4) if isinstance(value, float) else value)

            #y_pred_optuna = best_optuna.predict(x2_test)

            mse_optuna = mean_squared_error(y_test, y_pred_optuna)
            rmse_optuna = np.sqrt(mse_optuna)
            r2_optuna = r2_score(y_test, y_pred_optuna)

            evalute_model(best_optuna, x2_test, y_test, y_pred_optuna, "LightGBM Regressor With optuna", "Overall Qual" )

            #exponencional
            y_pred_optuna = np.expm1(y_pred_optuna)  # Exponencial inversa de la predicción
            y_test_optuna = np.expm1(y_test)
            evalute_model(best_optuna, x2_test, y_test_optuna, y_pred_optuna, "LightGBM Regressor with optuna without exponetional", "Overall Qual" )
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        #+++++++++++++++++++++++++++++++++++++++++++++++++++ 3.1.3 LGBM Regressor with Early Stopping +++++++++++++++++++++++++++++++++++++++++++
        elif selected_option == "📉 LGBM Regressor with Early Stopping📉":
            gbm2 = lgb.LGBMRegressor(objective = 'regression', random_state = 42, n_estimators=1000) 
            gbm2.fit(x2_train, y_train, eval_set = [(x2_test, y_test)], eval_metric = 'rmse', callbacks = [early_stopping(stopping_rounds = 50), log_evaluation(0)])
            
            params_to_display = ["n_estimators", "max_depth", "learning_rate", "min_child_samples"]
            cols = st.columns(len(params_to_display))

            for col, param in zip(cols, params_to_display):
                value = gbm2.get_params()[param]
                col.metric(label=param, value=round(value, 4) if isinstance(value, float) else value)

            y_pred_early = gbm2.predict(x2_test)

            mse_early = mean_squared_error(y_test, y_pred_early)
            rmse_early = np.sqrt(mse_early)
            r2_early = r2_score(y_test, y_pred_early)

            evalute_model(gbm2, x2_test, y_test, y_pred_early, "LightGBM Regressor With Early Stopping ", "Overall Qual" )

            #exponencional
            y_pred_earlyStopping = np.expm1(y_pred_early)  # Exponencial inversa de la predicción
            y_test_earlyStopping = np.expm1(y_test)
            evalute_model(gbm2, x2_test, y_test_earlyStopping, y_pred_earlyStopping, "LightGBM Regressor with early stopping without exponetional", "Overall Qual" )
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ########################################################################################################################################
    with col2: 

        ###################################### 3.2 Grafica (aun nose de que pero quiero agregar una grafica) ################################### #PENDIENTE
        graph = pd.DataFrame(
            np.random.randn(100, 2),
            columns=[ "SalePrice", "Overall Qual"])
        st.line_chart(graph)
        ########################################################################################################################################

    with col3:

        ################################################## 3.3 Seccion cantidad de Overall Quall ###############################################
        # Slider para Overall Qual
        st.markdown("<h3 style='font-size: 24px; margin: 0; padding: 0; text-align: center;'>Selección Cantidad de Overall Qual</h3>", unsafe_allow_html=True)
        overall_qual_value = st.slider("", 1, 10, 5)
        
        # Mostrar valor
        st.markdown(f"""
            <p style='font-size: 20px; text-align: center;'>Valor actual de Overall Qual: <strong>{overall_qual_value}</strong></p>
        """, unsafe_allow_html=True)
        
        # Crear datos con columnas esperadas
        #new_data = pd.DataFrame([{col: 0 for col in expected_columns}])
        new_data['Overall Qual'] = overall_qual_value

        # Predicción
        prediction_log = model.predict(new_data)

        # Preparar predicción para desnormalizar
        predicted_df = pd.DataFrame(prediction_log, columns=['SalePrice_log'])
        predicted_df['Gr Liv Area_log'] = 0  # columna ficticia

        # Desnormalización e inversión de log
        predicted_rescaled = scaler.inverse_transform(predicted_df)
        prediction_rescaled = predicted_rescaled[:, 0]
        prediction = np.expm1(prediction_rescaled)
        prediction_before = prediction
        #st.metric("💰 Predicción final en dólares", f"${round(prediction[0], 2):,.2f}")
        st.markdown("<h3 style='font-size: 26px; text-align: center;'>Predicción final en dólares</h3>", unsafe_allow_html=True)
        # Centrar la métrica con markdown
        st.markdown("<div style='display: flex; justify-content: center;'><div>" 
                    f"<p style='font-size: 26px; font-weight: bold;'>${round(prediction[0], 2):,.2f}</p>"
                    "</div></div>", unsafe_allow_html=True)
        ########################################################################################################################################

    col4, col5 = st.columns([0.7,1])
    with col4:

        ################################################## 3.4 Top 10 Peores y mejor Resultados ###############################################
        st.markdown("""
            <style>
                .stHorizontal > div {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
            </style>
        """, unsafe_allow_html=True)

        col44, col55= st.columns([1,1])
        with col44:

            #+++++++++++++++++++++++++++++++++++++++++++++++++ 3.4.1 Top 10 Peores Predicciones ++++++++++++++++++++++++++++++++++++++++++++++++
            # Cargar modelo, columnas esperadas y scaler
            model, expected_columns, scaler = load_model()  # Esta función debe devolver el modelo, las columnas en orden, y el scaler entrenado

            # Seleccionar columnas de entrada para test
            x_test = df[expected_columns].copy()

            # Hacer predicción en log (SalePrice_log)
            prediction_log = model.predict(x_test)

            # Construir DataFrame con las mismas columnas que usó el scaler
            # Asegúrate que el scaler fue fit con ['Gr Liv Area_log', 'SalePrice_log'] en ese orden (o el orden que corresponda)
            predicted_df = pd.DataFrame({
                'Gr Liv Area_log': np.zeros(len(prediction_log)),  # Placeholder
                'SalePrice_log': prediction_log
            })[scaler.feature_names_in_]  # Garantiza el orden correcto de columnas

            # Desnormalizar
            predicted_rescaled = scaler.inverse_transform(predicted_df)
            prediction_final = np.expm1(predicted_rescaled[:, list(scaler.feature_names_in_).index('SalePrice_log')])

            # Comparar con valores reales
            real_values = df['SalePrice'].values
            df_comparison = pd.DataFrame({
                'Real': real_values,
                'Predicción': prediction_final
            })
            df_comparison['Error absoluto'] = np.abs(df_comparison['Real'] - df_comparison['Predicción'])

            # Mostrar peores resultados
            st.markdown("<h3 style='font-size: 24px; margin: 0; padding: 0; text-align: center;'>Top 10 Peores resultados<br></h3>", unsafe_allow_html=True)
            errores_mayores = df_comparison.sort_values('Error absoluto', ascending=False).head(10)
            st.write(errores_mayores)
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with col55:

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 3.4.1 Top 10 Mejores Predicciones +++++++++++++++++++++++++++++++++++++++++++++
            # Mostrar mejores resultados
            st.markdown("<div style='display: flex; justify-content: center;'><div style='text-align: center;'>"
                        "<h3 style='font-size: 24px; margin: 0; padding: 0;'>Top 10 Mejores resultados<br></h3>"
                        "</div></div>", unsafe_allow_html=True)
            mejores_predicciones = df_comparison.sort_values('Error absoluto', ascending=True).head(10)
            st.write(mejores_predicciones)
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #########################################################################################################################################
    
    with col5:
    
        ######################################################## 4.Seccion de desglose informacion ##################################################
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++ 4.1 Valor ingresado de Overall Qual +++++++++++++++++++++++++++++++++++++++++++++++++
        st.markdown(
            f"""
            <div style="text-align: center; font-size: 28px;">
                <strong>🔧 Valor ingresado de Overall Qual 🔧</strong> <strong> <br>{overall_qual_value}</strong> 
            </div>
            """,
            unsafe_allow_html=True
        )

        subcol1, subcol2, subcol3 = st.columns([1,1,1])
        with subcol1:

            #++++++++++++++++++++++++++++++++++++++++++++++++ 4.1 Predicciones en dolares y valor real ++++++++++++++++++++++++++++++++++++++++++++++++
            st.markdown(
                """
                <div style="text-align: center;">
                    <h4>💰 Predicción en dólares 💰</h4>
                    <p style="font-size: 30px;"><strong>${:,.2f}</strong></p>
                </div>
                """.format(round(prediction[0], 2)),
                unsafe_allow_html=True
            )
            # Filtrar para obtener una fila con ese Overall Qual (puedes usar la más cercana si no hay exacta)
            fila_real = df.loc[df['Overall Qual'] == overall_qual_value]

            if not fila_real.empty:
                valor_real = fila_real.iloc[0]['SalePrice']
            else:
                valor_real = None

            st.markdown(
                """
                <div style="text-align: center;">
                    <h4>🏠 Valor real (cercano) 🏠 </h4>
                    <p style="font-size: 30px;"><strong>${:,.2f}</strong></p>
                </div>
                """.format(valor_real if valor_real is not None else 0),
                unsafe_allow_html=True
            )

            # Calcular error
            error_abs = abs(valor_real - prediction[0])
            error_pct = (error_abs / valor_real) * 100
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            

        with subcol2:
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++ 4.2 Error absoluto y Error Porcentual ++++++++++++++++++++++++++++++++++++++++++++++++++
            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <h4>📉 Error absoluto 📉</h4>
                    <p style='font-size: 30px;'><strong>${round(error_abs, 2):,.2f}</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style='text-align: center;'>
                  <h4> 📊 Error porcentual 📊 </h4>
                  <p style='font-size: 30px;'><strong>{round(error_pct, 2)}%</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        
        with subcol3:

            #++++++++++++++++++++++++++++++++++++++++++++++++++ 4.2 Error absoluto y Error Porcentual ++++++++++++++++++++++++++++++++++++++++++++++++++
            # Determinar color y mensaje según error
            if error_pct < 10:
                color = "green"
                mensaje = "✅ Excelente predicción ✅"
            elif error_pct < 25:
                color = "blue"
                mensaje = "🟦 Buena predicción 🟦"
            elif error_pct < 50:
                color = "orange"
                mensaje = "🟨 Regular 🟨"
            else:
                color = "red"
                mensaje = "🟥 Mala predicción 🟥"

            st.markdown(
                f"""
                <div style="text-align: center; font-size: 24px;">
                    {mensaje}
                </div>
                """,
                unsafe_allow_html=True
            )

            donut_chart = make_donut(error_pct, "Error", color)
            st.altair_chart(donut_chart)
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ################################################################################################################################################


elif section == "Development":
        #Verificar dimensiones 
    df_rows, df_columns = df.shape
    ######################################################## 5. Basic Information and First 5 Datas ######################################################
    col1, col2 = st.columns([0.25,2])
    with col1:
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 5.1 Basic Information ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        st.markdown("<h3 style='text-align: center;'>📌 Basic Info</h3>", unsafe_allow_html=True)
        st.markdown(f"**Number of rows:** {df_rows}<br>**Number of columns:** {df_columns}", unsafe_allow_html=True)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    with col2:
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 5.2 First 5 Datas +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        st.markdown("<h3 style='text-align: center;'>First 5 datas</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ######################################################################################################################################################

    #Crear dos columnas: izquierda(info) y derecha(EDA)
    #Ajusta proporcion si quieres mas espacio para el heatmap
    col1, col2, col3 = st.columns([1.2, 2, 2])
    #COLUMNA IZQUIERDA:INFO
    #3.1 Mostrar mas informacion
    with col1:

        ############################################################## 6.MORE INFORMATION ################################################################
        def show_centered_table(df_table, title=None):
            # Mostrar el título solo si se especifica y no está vacío
            if title and title.strip():
                st.markdown(f"<h3 style='text-align: center; margin-bottom: 0px;'>{title}</h3>", unsafe_allow_html=True)

            # Convertir el dataframe en HTML
            html_table = df_table.to_html(index=False)

            # Estilos personalizados para la tabla con colores adaptativos y bordes
            styled_html = f"""
            <html>
            <head>
                <style>
                    body {{
                        background-color: transparent;
                        color: inherit;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: auto;
                        font-family: sans-serif;
                        background-color: transparent;
                        color: inherit;
                    }}
                    th, td {{
                        text-align: center;
                        padding: 8px;
                        border: 1px solid #3c3e43;
                    }}
                    th {{
                        background-color: #1a1c24;
                        color: #8f9da4;
                        font-weight: bold;
                    }}
                    td {{
                        color: #fafafa;
                    }}
                    td:first-child {{
                        color: #8f9da4;
                    }}
                    tr:nth-child(odd) td:first-child,
                    tr:nth-child(even) td:first-child {{
                        background-color: inherit;
                    }}
                </style>
            </head>
            <body>
                {html_table}
            </body>
            </html>
            """

            # Renderizar el HTML con la tabla estilizada
            components.html(styled_html, height=400, scrolling=True)

        # ----------------------
        # OPCIONES DE VISUALIZACIÓN
        # ----------------------
        options = [
            "📊 Top 10 Correlated 📊", "❌ Null Values ❌", "📉 Descriptive Statistics 📉",
            "🔢 First Normalized Values 🔢", "🧪 Shapiro-Wilk Test 🧪", "📐 Skew Values 📐"
        ]

        selected_option = st.selectbox("", options)

        # ----------------------
        # FUNCIÓN PARA SHAPIRO-WILK
        # ----------------------
        @st.cache_data
        def get_shapiro_results(df):
            stat1, p1 = shapiro(df['SalePrice'])
            stat2, p2 = shapiro(df['Gr Liv Area'])
            stat3, p3 = shapiro(df['Overall Qual'])
            return pd.DataFrame({
                "Feature": ["SalePrice", "Gr Liv Area", "Overall Qual"],
                "Stat": [round(stat1, 4), round(stat2, 4), round(stat3, 4)],
                "P-value": [f"{p1:.2e}", f"{p2:.2e}", f"{p3:.2e}"]
            })

        # ----------------------
        # VISUALIZACIONES
        # ----------------------
                
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 6.1 Top 10 Correlated +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if selected_option == "📊 Top 10 Correlated 📊":
            saleprice_corr = df_corr['SalePrice'].abs().sort_values(ascending=False).head(10).reset_index()
            saleprice_corr.columns = ["Feature", "Correlation"]
            show_centered_table(saleprice_corr)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 6.2 Null Values ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif selected_option == "❌ Null Values ❌":
            null_table = df[['SalePrice', 'Gr Liv Area', 'Overall Qual']].isnull().sum().to_frame(name="Null Count").reset_index()
            null_table.columns = ["Feature", "Null Count"]
            show_centered_table(null_table)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 6.3 Descriptive Statistics ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif selected_option == "📉 Descriptive Statistics 📉":
            desc_stats = df[['Gr Liv Area', 'SalePrice', 'Overall Qual']].describe().reset_index()
            show_centered_table(desc_stats)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 6.4 First Normalized Values +++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif selected_option == "🔢 First Normalized Values 🔢":
            df[['SalePrice_log', 'Gr Liv Area_log']] = scaler.fit_transform(df[['SalePrice_log', 'Gr Liv Area_log']])
            show_centered_table(df[['SalePrice_log', 'Gr Liv Area_log']].head().reset_index(drop=True))
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 6.4 Shapiro-Wilk Test +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif selected_option == "🧪 Shapiro-Wilk Test 🧪":
            shapiro_table = get_shapiro_results(df)
            show_centered_table(shapiro_table)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 6.5 Skew Values ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif selected_option == "📐 Skew Values 📐":
            skew_table = pd.DataFrame({
                'Feature': ['SalePrice', 'Gr Liv Area', 'Overall Qual'],
                'Skewness': [
                    df['SalePrice'].skew(),
                    df['Gr Liv Area'].skew(),
                    df['Overall Qual'].skew()
                ]
            })
            show_centered_table(skew_table)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ######################################################################################################################################################

    with col2:
        ################################################################ 6.Heat Map #####3################################################################
        st.markdown("<h3 style='text-align: center;'>Heat Map</h3>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df_encoded[selected_features].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
        ax.set_title('Correlation Matrix of Selected Features')
        st.pyplot(fig)
        
            # Asegurarse de que estamos trabajando con arrays unidimensionales
        gr_liv_area_data = df['Gr Liv Area'].dropna().to_numpy().flatten()
        sale_price_data = df['SalePrice'].dropna().to_numpy().flatten()
        overall_qual_data = df['Overall Qual'].dropna().to_numpy().flatten()

        # Asegurarse de que estamos trabajando con arrays unidimensionales
        gr_liv_area_data = df['Gr Liv Area'].dropna().to_numpy().flatten()
        sale_price_data = df['SalePrice'].dropna().to_numpy().flatten()
        overall_qual_data = df['Overall Qual'].dropna().to_numpy().flatten()
        ##################################################################################################################################################

    with col3:

        ################################################################ 7.TYPES OF PLOTS ################################################################
    # Selector
        graph_option = st.selectbox("",["📦 Box Plots 📦", "📊 Histogram 📊"])

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 7.1 Box Plots ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if graph_option == "📦 Box Plots 📦":
            #st.markdown("<h3 style='text-align: center;'>Box Plots</h3>", unsafe_allow_html=True)
            fig, axs = plt.subplots(1, 3, figsize=(21, 7))

            sns.boxplot(y=df['Gr Liv Area'], ax=axs[0], color='skyblue')
            axs[0].set_title('Boxplot of Gr Liv Area')

            sns.boxplot(y=df['SalePrice'], ax=axs[1], color='skyblue')
            axs[1].set_title('Boxplot of SalePrice')

            sns.boxplot(y=df['Overall Qual'], ax=axs[2], color='skyblue')
            axs[2].set_title('Boxplot of Overall Qual')

            st.pyplot(fig)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 7.2 Histogram ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif graph_option == "📊 Histogram 📊":
            #st.markdown("<h3 style='text-align: center;'>Basic Histogram</h3>", unsafe_allow_html=True)

            # Datos
            gr_liv_area_data = df['Gr Liv Area'].dropna()
            sale_price_data = df['SalePrice'].dropna()
            overall_qual_data = df['Overall Qual'].dropna()

            fig3, axs = plt.subplots(1, 3, figsize=(18, 6))

            # Histograma con KDE - Gr Liv Area
            axs[0].hist(gr_liv_area_data, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
            kde = gaussian_kde(gr_liv_area_data)
            x_vals = np.linspace(gr_liv_area_data.min(), gr_liv_area_data.max(), 500)
            axs[0].plot(x_vals, kde(x_vals), color='blue', linewidth=2, label='KDE')
            axs[0].set_title("Distribution of Gr Liv Area with KDE")
            axs[0].legend()

            # Histograma con KDE - SalePrice
            axs[1].hist(sale_price_data, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
            kde = gaussian_kde(sale_price_data)
            x_vals = np.linspace(sale_price_data.min(), sale_price_data.max(), 500)
            axs[1].plot(x_vals, kde(x_vals), color='blue', linewidth=2, label='KDE')
            axs[1].set_title("Distribution of SalePrice with KDE")
            axs[1].legend()

            # Histograma discreto - Overall Qual
            sns.histplot(overall_qual_data, discrete=True, ax=axs[2], color='skyblue')
            axs[2].set_title("Distribution of Overall Quality")

            st.pyplot(fig3)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ##################################################################################################################################################

    ############################################## 8. Histogram with log1p and Histogram with log1p + MinMaxScaler #######################################
    # Opción de transformación
        section = st.radio(
            label="",
            options=["📈 Histogram with log1p 📈", "📉 Histogram with log1p + MinMaxScaler 📉"],
            horizontal=True
        )

        #transformation_option = st.selectbox("", ["📈 Histogram with log1p 📈", "📉 Histogram with log1p + MinMaxScaler 📉"])
        # Datos necesarios
        overall_qual_data = df['Overall Qual'].dropna()
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 8.2 Histogram with log1p +++++++++++++++++++++++++++++++++++++++++++++++++++++
        if section == "📈 Histogram with log1p 📈":
        #if transformation_option == "📈 Histogram with log1p 📈":
            #st.markdown("<h3 style='text-align: center;'>Histogram with log1p</h3>", unsafe_allow_html=True)

            df['SalePrice_log'] = np.log1p(df['SalePrice'])
            df['Gr Liv Area_log'] = np.log1p(df['Gr Liv Area'])

            saleprice_log_data = df['SalePrice_log'].dropna().to_numpy().flatten()
            grlivarea_log_data = df['Gr Liv Area_log'].dropna().to_numpy().flatten()

            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            axs[0].hist(saleprice_log_data, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
            kde_sp_log = gaussian_kde(saleprice_log_data)
            x_sp_log = np.linspace(saleprice_log_data.min(), saleprice_log_data.max(), 500)
            axs[0].plot(x_sp_log, kde_sp_log(x_sp_log), color='blue', linewidth=2, label='KDE')
            axs[0].set_title("Distribution of SalePrice after log")
            axs[0].set_xlabel("SalePrice (log1p)")
            axs[0].set_ylabel("Density")
            axs[0].legend()

            axs[1].hist(grlivarea_log_data, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
            kde_la_log = gaussian_kde(grlivarea_log_data)
            x_la_log = np.linspace(grlivarea_log_data.min(), grlivarea_log_data.max(), 500)
            axs[1].plot(x_la_log, kde_la_log(x_la_log), color='blue', linewidth=2, label='KDE')
            axs[1].set_title("Distribution of Gr Liv Area after log")
            axs[1].set_xlabel("Gr Liv Area (log1p)")
            axs[1].set_ylabel("Density")
            axs[1].legend()

            sns.histplot(overall_qual_data, discrete=True, ax=axs[2], color='skyblue', edgecolor='black')
            axs[2].set_title("Distribution of Overall Qual")
            axs[2].set_xlabel("Overall Qual")
            axs[2].set_ylabel("Count")

            st.pyplot(fig)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 8.3 Histogram with log1p + MinMaxScaler +++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif section == "📉 Histogram with log1p + MinMaxScaler 📉": 
            # Asegurar que las columnas estén disponibles
            if 'SalePrice_log' not in df.columns or 'Gr Liv Area_log' not in df.columns:
                df['SalePrice_log'] = np.log1p(df['SalePrice'])
                df['Gr Liv Area_log'] = np.log1p(df['Gr Liv Area'])

            scaler = MinMaxScaler()
            df[['SalePrice_log', 'Gr Liv Area_log']] = scaler.fit_transform(df[['SalePrice_log', 'Gr Liv Area_log']])

            saleprice_scaled = df['SalePrice_log'].dropna().to_numpy().flatten()
            grlivarea_scaled = df['Gr Liv Area_log'].dropna().to_numpy().flatten()

            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            axs[0].hist(saleprice_scaled, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
            kde1 = gaussian_kde(saleprice_scaled)
            x_vals1 = np.linspace(saleprice_scaled.min(), saleprice_scaled.max(), 500)
            axs[0].plot(x_vals1, kde1(x_vals1), color='blue', linewidth=2, label='KDE')
            axs[0].set_title("SalePrice after Log + MinMaxScaler")
            axs[0].set_xlabel("Scaled Value")
            axs[0].set_ylabel("Density")
            axs[0].legend()

            axs[1].hist(grlivarea_scaled, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
            kde2 = gaussian_kde(grlivarea_scaled)
            x_vals2 = np.linspace(grlivarea_scaled.min(), grlivarea_scaled.max(), 500)
            axs[1].plot(x_vals2, kde2(x_vals2), color='blue', linewidth=2, label='KDE')
            axs[1].set_title("Gr Liv Area after Log + MinMaxScaler")
            axs[1].set_xlabel("Scaled Value")
            axs[1].set_ylabel("Density")
            axs[1].legend()

            sns.histplot(overall_qual_data, discrete=True, ax=axs[2], color='skyblue', edgecolor='black')
            axs[2].set_title("Distribution of Overall Qual")
            axs[2].set_xlabel("Overall Qual")
            axs[2].set_ylabel("Count")

            st.pyplot(fig)
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ##################################################################################################################################################
#----------------------------------------------------------------------------------------------------------------------------#
