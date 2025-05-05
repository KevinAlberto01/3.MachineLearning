#0.IMPORT LIBRARIES 
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler ##
import streamlit.components.v1 as components
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import early_stopping, log_evaluation 
#--------------------------------- 1.PAGE CONFIGURATION ---------------------------------#
# Configurar página en modo ancho
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")
# Titulo de la pagina
st.markdown("<h1 style = 'text-align: center;'>🏠 House Price Prediction 🏠</h1>", unsafe_allow_html=True)
#Titulos de seleccion 

# Crear columnas vacías a los lados para centrar el radio
col1, col2, col3 = st.columns([3, 1.3, 3])  # Ajusta proporciones si quieres

with col2:
    section = st.radio(
        label="",
        options=["Basic", "Development", "Advanced", "Results"],
        horizontal=True
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
#-----------------------------------------------------------------------------------------#

#--------------------------------- 2.DEVELOPMENT ---------------------------------#

#1.LOAD DATASET
file_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/AmesHousing.csv'
df = pd.read_csv(file_path)
selected_features = ['SalePrice', 'Overall Qual', 'Gr Liv Area', 'Garage Cars', 
                     'Garage Area', 'Total Bsmt SF', '1st Flr SF', 'Full Bath', 'Year Built']
df_encoded = pd.get_dummies(df, drop_first=True)
df_corr = df_encoded.corr()

df['SalePrice_log'] = np.log1p(df['SalePrice'])
df['Gr Liv Area_log'] = np.log1p(df['Gr Liv Area'])
scaler = MinMaxScaler()

def evalute_model(model, x_test, y_test, y_pred, model_name, feature):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Model: {model_name} for {feature}")
    st.write(f"MSE: {mse:.4f}")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"R2: {r2:.4f}")
    st.write("-" * 50)


#---------------------------------------------------------------------------------#

#---------------------------------- 3.MAIN ---------------------------------#

if section == "Basic":
    y = df['SalePrice_log']
    x2 = df[['Overall Qual']]
    x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, random_state = 42)

    gbm2 = lgb.LGBMRegressor(objective = 'regression', random_state = 42, verbosity = -1)
    gbm2.fit(x2_train, y_train)
    y_pred = gbm2.predict(x2_test)

    mse = mean_squared_error(y_test, y_pred)
    st.write("Evaluation of Development LGBM Regressor")
    st.write(f"📊 **MSE (Overall Qual):** `{mse:.4f}`")
    evalute_model(gbm2, x2_test, y_test, y_pred, "LightGBM Regressor", "Overall Qual" )

elif section == "Development":
    #---------------------------------- 2.SIDEBAR ---------------------------------#
    #Verificar dimensiones 
    df_rows, df_columns = df.shape
    col1, col2 = st.columns([0.25,2])
    with col1:
        st.markdown("<h3 style='text-align: center;'>📌 Basic Info</h3>", unsafe_allow_html=True)
        st.markdown(f"**Number of rows:** {df_rows}<br>**Number of columns:** {df_columns}", unsafe_allow_html=True)
    #------------------------------------------------------------------------------#
    
    with col2:
        
        st.markdown("<h3 style='text-align: center;'>First 5 datas</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)

    #Crear dos columnas: izquierda(info) y derecha(EDA)
    #Ajusta proporcion si quieres mas espacio para el heatmap
    col1, col2, col3 = st.columns([1.2, 2, 2])
    #COLUMNA IZQUIERDA:INFO
    #3.1 Mostrar mas informacion
    with col1:
        # ----------------------
        # FUNCIÓN PARA MOSTRAR TABLAS CENTRADAS
        # ----------------------

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
        if selected_option == "📊 Top 10 Correlated 📊":
            saleprice_corr = df_corr['SalePrice'].abs().sort_values(ascending=False).head(10).reset_index()
            saleprice_corr.columns = ["Feature", "Correlation"]
            show_centered_table(saleprice_corr)

        elif selected_option == "❌ Null Values ❌":
            null_table = df[['SalePrice', 'Gr Liv Area', 'Overall Qual']].isnull().sum().to_frame(name="Null Count").reset_index()
            null_table.columns = ["Feature", "Null Count"]
            show_centered_table(null_table)

        elif selected_option == "📉 Descriptive Statistics 📉":
            desc_stats = df[['Gr Liv Area', 'SalePrice', 'Overall Qual']].describe().reset_index()
            show_centered_table(desc_stats)

        elif selected_option == "🔢 First Normalized Values 🔢":
            #df['SalePrice_log'] = np.log1p(df['SalePrice'])
            #df['Gr Liv Area_log'] = np.log1p(df['Gr Liv Area'])
            #scaler = MinMaxScaler()
            df[['SalePrice_log', 'Gr Liv Area_log']] = scaler.fit_transform(df[['SalePrice_log', 'Gr Liv Area_log']])
            show_centered_table(df[['SalePrice_log', 'Gr Liv Area_log']].head().reset_index(drop=True))

        elif selected_option == "🧪 Shapiro-Wilk Test 🧪":
            shapiro_table = get_shapiro_results(df)
            show_centered_table(shapiro_table)

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


    with col2:
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

    with col3:

    # Selector
        graph_option = st.selectbox("",["📦 Box Plots 📦", "📊 Histogram 📊"])

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


    # Opción de transformación
        transformation_option = st.selectbox("", ["📈 Histogram with log1p 📈", "📉 Histogram with log1p + MinMaxScaler 📉"])

        # Datos necesarios
        overall_qual_data = df['Overall Qual'].dropna()

        if transformation_option == "📈 Histogram with log1p 📈":
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

        elif transformation_option == "📉 Histogram with log1p + MinMaxScaler 📉":

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

    #---------------------------------------------------------------------------#