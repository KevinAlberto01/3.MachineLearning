import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(layout="wide")

# Título grande y centrado al inicio
st.markdown("<h1 style='text-align: center; font-size: 42px; font-weight: bold;'>DASHBOARD MNIST</h1>", unsafe_allow_html=True)

# Datos y modelo
digits = datasets.load_digits()
x = digits.data
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

param_grid = {'C': [1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=3)
grid_search.fit(x_train, y_train)

y_pred = grid_search.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

accuracy = np.mean(y_pred == y_test)

df_report = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'Class'})
df_report_reset = df_report.round(3)

class_counts = pd.DataFrame(np.unique(y, return_counts=True)).T
class_counts.columns = ['Class', 'Examples per class']
df_info_no_index = class_counts


# Función actualizada para el donut chart con el título arriba más pegado y el color azul más fuerte
def make_donut(input_response, input_color='blue'):
    if input_color == 'blue':
        chart_color = ['#E0F0FF', '#2A79B9']  # Azul claro y azul más fuerte
    else:
        chart_color = ['#E0F0FF', '#2A79B9']

    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=100)

    # Fondo transparente
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    # Título arriba, más grande y más pegado - en blanco
    ax.text(0.5, 0.97, "Model Accuracy", ha='center', va='center',
            fontsize=16, fontweight='bold', color='white', transform=ax.transAxes)

    ax.pie([input_response, 100 - input_response],
           colors=[chart_color[1], chart_color[0]],
           startangle=90,
           counterclock=False,
           wedgeprops=dict(width=0.26, edgecolor='white'))

    # Porcentaje en el centro, subido un poco, también en blanco
    ax.text(0, 0.06, f'{input_response:.1f}%', ha='center', va='center',
            fontsize=24, fontweight='bold', color='white')

    ax.set(aspect="equal")
    plt.close(fig)
    return fig




# Streamlit Layout
col1_top, col2_top, col3_top = st.columns([0.7, 3, 1.2])

with col1_top:
    st.markdown("<h3 style='text-align: center;'>Data Info</h3>", unsafe_allow_html=True)
    info_text = f"""
    **Dimension of x:** {x.shape}  
    **Dimension of y:** {y.shape}  
    """
    st.markdown(info_text)

    fig_donut = make_donut(accuracy * 100)
    st.pyplot(fig_donut)

with col2_top:
    st.markdown("<h3 style='text-align: center;'>True vs. Predicted</h3>", unsafe_allow_html=True)
    fig_images, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    for i, ax in enumerate(axes[:10]):
        img = x_test[i].reshape(8, 8)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {y_test[i]} - Predicted: {y_pred[i]}", fontsize=10)
        ax.axis('off')
    plt.subplots_adjust(hspace=0.3)
    st.pyplot(fig_images)

with col3_top:
    st.markdown("<h3 style='text-align: center;'>Confusion Matrix</h3>", unsafe_allow_html=True)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 5), dpi=300)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=digits.target_names,
                yticklabels=digits.target_names, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    buf = io.BytesIO()
    fig_cm.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    st.image(Image.open(buf), width=400)

col1_bot, col2_bot, col3_bot, col4_bot = st.columns([0.8, 0.8, 2.2, 1.4])

with col1_bot:
    st.markdown("<h3 style='text-align: center;'>Class Distribution</h3>", unsafe_allow_html=True)
    st.markdown("<div style='max-width: 350px; overflow-x: auto;'>", unsafe_allow_html=True)
    st.markdown(df_info_no_index.to_html(index=False), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2_bot:
    st.markdown("<h3 style='text-align: center;'>Best Parameters</h3>", unsafe_allow_html=True)
    best_params = grid_search.best_params_
    best_params_text = f"""
    **C:** {best_params['C']}  
    **Kernel:** {best_params['kernel']}  
    **Hyperparameter search space:**  
    """
    st.markdown(best_params_text)
    st.write(param_grid)

with col3_bot:
    st.markdown("<h3 style='text-align: center;'>Normalized Data</h3>", unsafe_allow_html=True)
    st.write(pd.DataFrame(x_train[:5], columns=[f"F{i}" for i in range(x_train.shape[1])]))

with col4_bot:
    st.markdown("<h3 style='text-align: center;'>Classification Report</h3>", unsafe_allow_html=True)
    st.markdown("<div style='width: 100%; overflow-x: auto;'>", unsafe_allow_html=True)
    st.markdown(df_report_reset.to_html(index=False), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
