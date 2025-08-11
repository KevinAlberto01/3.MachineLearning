#========================== 0.INICIO DE "LIBRERIAS" ==========================#
#import streamlit as st
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
#============================ 0.FIN DE "LIBRERIAS" ===========================#

#======================== 1.INICIO DE "MACHINE LEARNING" =====================#
#lectura de archivo
df = pd.read_csv("1.Basic/Steps/3.1DataProcessing/IMDB Dataset.csv")
print()
#df['br_count'] = df['review'].str.count(r'<br\s*/?>', flags=re.IGNORECASE)
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


#MATRIX 
#model#
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

modelo_cargado = joblib.load('1.Basic/Steps/3.6JoinAll(1-2)/LogisticPipeline.pkl')

def predecir_sentimiento(texto):
    label_map = {0: 'Negative', 1: 'Positive'}
    prediccion = modelo_cargado.predict([texto])
    probabilidades = modelo_cargado.predict_proba([texto])
    
    prediccion_texto = label_map[prediccion[0]]
    prob_pos = probabilidades[0][1]  # Probabilidad de clase positiva
    
    return prediccion_texto, prob_pos

#========================== 1.FIN DE "MACHINE LEARNING" ======================#

#============================ 2.INICIO DE "DASHBOARD" ========================#
#----------------------------- 2.1 INICIO - COLUMNA 1 ------------------------#

#///////////// 2.1.1 INICIO DE CONTEO TOTAL DE REVIEWS ANALIZADAS ////////////#
print(df.shape)
#/////////////// 2.1.1 FIN DE CONTEO TOTAL DE REVIEWS ANALIZADAS /////////////#

#///////////////// 2.1.2 INICIO DE PROMEDIO DE SENTIMIENTO GLOBAL ////////////#
print(df['sentiment'].value_counts())
#/////////////////// 2.1.2 FIN DE PROMEDIO DE SENTIMIENTO GLOBAL /////////////#

#////////////// 2.1.3 INICIO DE TOP 10 POSITIVAS Y NEGATIVAS /////////////////#
# Contar las palabras positivas
counter_pos = Counter(positive_words)
top10_pos = counter_pos.most_common(10)

# Contar las palabras negativas
counter_neg = Counter(negative_words)
top10_neg = counter_neg.most_common(10)

print("Top 10 palabras positivas:", top10_pos)
print("Top 10 palabras negativas:", top10_neg)
#//////////////// 2.1.3 FIN DE TOP 10 POSITIVAS Y NEGATIVAS //////////////////#

#------------------------------- 2.1 FIN - COLUMNA 1 ------------------------#

#----------------------------- 2.2 INICIO - COLUMNA 2 ------------------------#

#/////////////////////////// 2.2.1 INICIO DE WORLDCLOUD //////////////////////#
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.title('Palabras más comunes (Positivas)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.title('Palabras más comunes (Negativas)')
plt.axis('off')
plt.show()
#///////////////////////////// 2.2.1 FIN DE WORLDCLOUD ///////////////////////#

#///////////////////////////// 2.2.2 INICIO MUESTRAS 5 ///////////////////////#
print(df['review_clean'].head(5))
#/////////////////////////////// 2.2.2 FIN MUESTRAS 5 ////////////////////////#

#------------------------------- 2.2 FIN - COLUMNA 2 -------------------------#

#----------------------------- 2.3 INICIO - COLUMNA 3 ------------------------#

#//////////////////////// 2.2.1 INICIO DE MATRIX CONFUSION ///////////////////#
#print("Confusion Matrix: \n", confusion_matrix(y_test,predictions))
disp=ConfusionMatrixDisplay.from_predictions(y_test,predictions, cmap='Blues')
plt.show()
#////////////////////////// 2.2.1 FIN DE MATRIX CONFUSION ////////////////////#

#//////////////////////// 2.2.1 INICIO DE PREDICTIONS ////////////////////////#
# Lista de frases positivas y negativas combinadas
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

# Iterar y predecir
for texto in textos_prueba:
    sentimiento, probabilidad = predecir_sentimiento(texto)
    print(f"Texto: {texto}")
    print(f"Predicción: {sentimiento} | Probabilidad de positivo: {probabilidad:.2%}")
    print("-" * 50)
#///////////////////////// 2.2.1 FIN DE PREDICTIONS //////////////////////////#

#------------------------------- 2.3 FIN - COLUMNA 3 ------------------------#

#============================== 2.FIN DE "DASHBOARD" =========================#
