#========================== 0.INICIO DE "LIBRERIAS" ==========================#
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import joblib
import os
#============================ 0.FIN DE "LIBRERIAS" ===========================#

st.set_page_config(layout="wide", page_title="Sentiment Analysis Dashboard")

#======================== 1.INICIO DE "MACHINE LEARNING" =====================#
#------------------------- 1.1 INICIO LECTURA DE ARCHIVO ---------------------#
file_path = os.path.join(os.path.dirname(__file__), 'IMDBDataset.csv')
df = pd.read_csv(file_path)
#--------------------------- 1.1 FIN LECTURA DE ARCHIVO ----------------------#

#----------------------- 1.1 INICIO DE LIMPIEZA DE ARCHIVO -------------------#
df['review'] = df['review'].str.replace(r'<br\s*/?>', ' ', regex=True)

#comprobacion de como quedaron las columnas 
print(df['sentiment'].value_counts()) #Muestra cuantos hay por cada columna

# 3.4 LIMPIEZA DE TEXTO ADICIONAL
df.drop_duplicates(subset= 'review',inplace=True)

def limpiar_texto(texto):
    texto = texto.lower()  # minúsculas
    texto = re.sub(r'<.*?>', ' ', texto)  # etiquetas HTML restantes
    texto = re.sub(r'\d+', '', texto)  # eliminar números
    texto = texto.translate(str.maketrans('', '', string.punctuation))  # eliminar signos puntuación
    texto = re.sub(r'\s+', ' ', texto)  # eliminar espacios múltiples
    return texto.strip()
df['review_clean'] = df['review'].apply(limpiar_texto) #aqui se agrega una nueva columna pero es solamente la correccion
#------------------------ 1.1 FIN DE LIMPIEZA DE ARCHIVO ---------------------#

#---------------------------- 1.1 INICIO DE STOPWORDS ------------------------#
# 3.5 Eliminar stopwords personalizadas
# Stopwords en inglés + personalizadas
stop_words = list(text.ENGLISH_STOP_WORDS.union(stopwords.words('english')))
stop_words += ['movie', 'film', 'one', 'character', 'time', 'story', 'make', 'see',
    'scene', 'way', 'thing', 'look', 'plot', 'work', 'director', 'watch',
    'get', 'go', 'going', 'even', 'bit', 'really', 'know', 'think',
    'much', 'well', 'take', 'still', 'say', 'something', 'lot', 'back',
    'also', 'end', 'though', 'better', 'people', 'little', 'nothing',
    'makes', 'right', 'man', 'woman', 'new', 'life', 'im'
]
vectorizer = TfidfVectorizer(stop_words=stop_words,max_features=5000)

#world cloud#
df = df.dropna(subset=['review_clean'])
positive_text = ' '.join(df[df['sentiment'] == 'positive']['review_clean'])
negative_text = ' '.join(df[df['sentiment'] == 'negative']['review_clean'])
stop_words_set = set(stop_words)

# Limpiar las palabras para WordCloud (eliminar stopwords)
positive_words = [word for word in positive_text.split() if word not in stop_words_set]
negative_words = [word for word in negative_text.split() if word not in stop_words_set]

# Volver a unir el texto limpio
positive_text_cleaned = ' '.join(positive_words)
negative_text_cleaned = ' '.join(negative_words)

wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text_cleaned)
wordcloud_neg = WordCloud(width=800, height=400, background_color='black').generate(negative_text_cleaned)
#------------------------------ 1.1 FIN DE STOPWORDS ------------------------#

#------------------------ 1.1 INICIO DE MATRIX CONFUSION --------------------#
#5.TRANSFORMACION DE VARIABLES CATEGORICAS (PARA NLP)
df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})
print(df['sentiment'])

#1.INICIO PREPARACION DE DATOS
x = vectorizer.fit_transform(df['review'])
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

modelL = LogisticRegression(C=1, solver='liblinear', random_state=42)
modelL.fit(X_train, y_train)
predictions = modelL.predict(X_test)
#------------------------- 1.1 FIN DE MATRIX CONFUSION ----------------------#

#------------------------- 1.1 INICIO DE MODELO CARGADO ---------------------#
# --- Función para cargar modelo y vectorizador (cached) ---
@st.cache_resource
def cargar_modelo_y_vectorizador():
    modelo_cargado = joblib.load(os.path.join(os.path.dirname(__file__), 'LogisticPipeline.pkl'))
    # En tu código parece que el vectorizador lo defines aparte; si usas pipeline, el vectorizador viene con el modelo.
    return modelo_cargado

def predecir_sentimiento(texto,modelo):
    label_map = {0: 'Negative', 1: 'Positive'}
    prediccion = modelo.predict([texto])
    probabilidades = modelo.predict_proba([texto])
    
    prediccion_texto = label_map[prediccion[0]]
    prob_pos = probabilidades[0][1]  # Probabilidad de clase positiva
    
    return prediccion_texto, prob_pos
#--------------------------- 1.1 FIN DE MODELO CARGADO ----------------------#

#========================== 1.FIN DE "MACHINE LEARNING" ======================#

#===================== 2.INICIO DE CONFIGURACION "STREAMLIT" =================#
st.markdown("<h1 style='text-align: center;'>😊 Sentiment Analysis 😊</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 3, 1])  # col2 será 3 veces más ancha que col1 y col3
#======================= 2.FIN DE CONFIGURACION "STREAMLIT" ==================#



#============================ 3.INICIO DE "DASHBOARD" ========================#

#----------------------------- 2.1 INICIO - COLUMNA 1 ------------------------#
with col1:
    st.markdown("<h2 style='text-align: center;'>Datos Generales</h2>", unsafe_allow_html=True)
    #///////////// 2.1.1 INICIO DE CONTEO TOTAL DE REVIEWS ANALIZADAS ////////////#
    st.markdown(f"<p style='text-align: center;'><strong>Total de reviews analizadas:</strong> {df.shape[0]}</p>", unsafe_allow_html=True)

    #/////////////// 2.1.1 FIN DE CONTEO TOTAL DE REVIEWS ANALIZADAS /////////////#

    #///////////////// 2.1.2 INICIO DE PROMEDIO DE SENTIMIENTO GLOBAL ////////////#
    st.write("Distribucion de sentimientos:")

    counts = df['sentiment'].value_counts().reset_index()
    counts.columns = ['Sentiment', 'Count']
    counts['Sentiment'] = counts['Sentiment'].map({1: "Positive", 0: "Negative"})

    st.table(counts)
    #/////////////////// 2.1.2 FIN DE PROMEDIO DE SENTIMIENTO GLOBAL /////////////#

    #////////////// 2.1.3 INICIO DE TOP 10 POSITIVAS Y NEGATIVAS /////////////////#
    # Contar las palabras positivas
    counter_pos = Counter(positive_words)
    top10_pos = counter_pos.most_common(4)
    st.markdown("<h3 style='text-align: center;'>Top 4 palabras positivas</h3>", unsafe_allow_html=True)
    st.table(top10_pos)

    # Contar las palabras negativas
    counter_neg = Counter(negative_words)
    top10_neg = counter_neg.most_common(4)
    st.markdown("<h3 style='text-align: center;'>Top 4 palabras negativas</h3>", unsafe_allow_html=True)
    st.table(top10_neg)

    #//////////////// 2.1.3 FIN DE TOP 10 POSITIVAS Y NEGATIVAS //////////////////#

#------------------------------- 2.1 FIN - COLUMNA 1 ------------------------#

with col2:
    #----------------------------- 2.2 INICIO - COLUMNA 2 ------------------------#

    #/////////////////////////// 2.2.1 INICIO DE WORLDCLOUD //////////////////////#
    st.markdown("<h2 style='text-align: center;'>WordCloud (Positivo vs Negativo)</h2>", unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].imshow(wordcloud_pos, interpolation = 'bilinear')
    ax[0].set_title('Palabras más comunes (Positivas)')
    ax[0].axis('off')

    ax[1].imshow(wordcloud_neg, interpolation = 'bilinear')
    ax[1].set_title('Palabras mas comunes (Negativas)')
    ax[1].axis('off')

    st.pyplot(fig)

    #///////////////////////////// 2.2.1 FIN DE WORLDCLOUD ///////////////////////#

    #///////////////////////////// 2.2.2 INICIO MUESTRAS 5 ///////////////////////#
    df['review_clean'] = df['review'].apply(limpiar_texto) #aqui se agrega una nueva columna pero es solamente la correccion
    st.markdown("<h2 style='text-align: center;'>Ejemplos de reviews limpias (5 primeras)</h2>", unsafe_allow_html=True)
    st.dataframe(df['review_clean'].head(5), width=1900) 

    #/////////////////////////////// 2.2.2 FIN MUESTRAS 5 ////////////////////////#

#------------------------------- 2.2 FIN - COLUMNA 2 -------------------------#

#----------------------------- 2.3 INICIO - COLUMNA 3 ------------------------#
with col3:

    #//////////////////////// 2.2.1 INICIO DE MATRIX CONFUSION ///////////////////#
    #print("Confusion Matrix: \n", confusion_matrix(y_test,predictions))
    st.markdown("<h2 style='text-align: center;'>Confusion Matrix</h2>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap='Blues', ax=ax)
    st.pyplot(fig)
    #////////////////////////// 2.2.1 FIN DE MATRIX CONFUSION ////////////////////#

    #//////////////////////// 2.2.1 INICIO DE PREDICTIONS ////////////////////////#
    modelo = cargar_modelo_y_vectorizador()
    # Lista de frases positivas y negativas combinadas
    # Lista de frases
    textos_prueba = [
        "This movie was fantastic! Loved every minute of it.",
        "Absolutely terrible film. Waste of time.",
        "The plot was engaging and the characters were lovable.",
        "I couldn’t stand watching it, boring and predictable.",
        "An unforgettable masterpiece with stunning visuals.",
        "Poor acting and weak storyline.",
        "I was pleasantly surprised by how good it was.",
        "The worst movie I've seen this year.",
        "Heartwarming and beautifully directed.",
        "Not worth the hype at all."
    ]

    # Selector para que el usuario elija una frase
    label_html = "<h2 style='text-align: center;'>Seleccionar texto</h2>"
    st.markdown(label_html, unsafe_allow_html=True)

    # Ponemos el selectbox sin label
    frase_seleccionada = st.selectbox("", textos_prueba, label_visibility="collapsed")
    
    # Llama a tu función de predicción
    sentimiento, probabilidad = predecir_sentimiento(frase_seleccionada,modelo)

    # Muestra resultado
    st.write(f"**Texto seleccionado:** {frase_seleccionada}")
    st.write(f"**Predicción:** {sentimiento}")

    # Preparar datos para barra horizontal
    prob_pos = probabilidad
    prob_neg = 1 - prob_pos

    labels = ['Positivo', 'Negativo']
    scores = [prob_pos, prob_neg]
    colors = ["#48BFF7", "#0090D3"]  # verde y rojo

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.barh(labels, scores, color=colors)
    ax.set_xlim(0, 1)
    for i, v in enumerate(scores):
        ax.text(v + 0.02, i, f"{v:.1%}", color='black', va='center')
    ax.set_xlabel('Probabilidad')
    ax.set_title('Probabilidad de Sentimiento')

    st.pyplot(fig)
    #///////////////////////// 2.2.1 FIN DE PREDICTIONS //////////////////////////#

    #------------------------------- 2.3 FIN - COLUMNA 3 ------------------------#
    

#============================== 3.FIN DE "DASHBOARD" =========================#
