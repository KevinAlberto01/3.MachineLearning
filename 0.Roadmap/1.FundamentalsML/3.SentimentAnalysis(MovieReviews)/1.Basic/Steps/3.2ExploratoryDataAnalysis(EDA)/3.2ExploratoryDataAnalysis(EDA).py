#0.LIBRERIAS 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import string
from sklearn.feature_extraction import text 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
#################### A.DATA PROCESSING ####################

#1.CARGA DE DATOS
df = pd.read_csv("1.Basic/Steps/3.1DataProcessing/IMDB Dataset.csv")
print()
print("Se leyo los datos correctamente")
print("LA BASE DE DATOS CUENTA CON UN TAMAÑO DE:",df.shape)
print()

#3.LIMPIEZA DE DATOS

#3.2 Elimina Duplicados
df.drop_duplicates(subset= 'review',inplace=True)
print("se elimino los datos duplicados")
print()

#3.3 Eliminacion de br (para NLP)
#df['br_count'] = df['review'].str.count(r'<br\s*/?>', flags=re.IGNORECASE)
df['review'] = df['review'].str.replace(r'<br\s*/?>', ' ', regex=True)

#comprobacion de como quedaron las columnas 
print(df['sentiment'].value_counts()) #Muestra cuantos hay por cada columna

# 3.4 LIMPIEZA DE TEXTO ADICIONAL
def limpiar_texto(texto):
    texto = texto.lower()  # minúsculas
    texto = re.sub(r'<.*?>', ' ', texto)  # etiquetas HTML restantes
    texto = re.sub(r'\d+', '', texto)  # eliminar números
    texto = texto.translate(str.maketrans('', '', string.punctuation))  # eliminar signos puntuación
    texto = re.sub(r'\s+', ' ', texto)  # eliminar espacios múltiples
    return texto.starip()

df['review_clean'] = df['review'].apply(limpiar_texto) #aqui se agrega una nueva columna pero es solamente la correccion
print("LA BASE DE DATOS AHORA TIENE UN TAMAÑO DE:",df.shape)
################# A.FIN DE DATA PROCESSING #################


'''
# Longitud ANTES de limpiar
df['length_original'] = df['review'].apply(lambda x: len(x.split()))
# Longitud DESPUÉS de limpiar
df['length_clean'] = df['review_clean'].apply(lambda x: len(x.split()))



#5.TRANSFORMACION DE VARIABLES CATEGORICAS (PARA NLP)
df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})
print(df['sentiment'])

print()
print(df.dtypes)
print() 
#Crear el vectorizador



vectorizer = TfidfVectorizer(stop_words='english',max_features=5000)
#Aplicar vectorizacion al texto 
x = vectorizer.fit_transform(df['review'])
#Etiquetas
y = df['sentiment']
print("Form of X:", x.shape)
print()

#FIN DEL PASO EXTRA POR SER NLP 

#7.DIVISION DE CONJUNTOS 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)
'''