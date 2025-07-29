#0.LIBRERIAS 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re

#################### A.DATA PROCESSING ####################
#1.Carga de datos 
df = pd.read_csv("1.Basic/Steps/3.1DataProcessing/IMDB Dataset.csv")
print()

#2.Inspeccion Inicial
'''print(df.shape) #tamaño de fila y columna
print()
print(df.info()) #Informacion de la base de datos 
print()
print(df.describe()) #Describe caracteristicas importantes
print()
print(df.head(10)) #Muestra las primeras 10 fila
print()
print(df['sentiment'].unique()) #Checa las categorias 
print()
print(df['sentiment'].value_counts()) #Muestra cuantos hay por cada columna
print()
print(df.dtypes) # Muestra el tipo de datos
'''

#3.Limpieza de datos 

#3.2 Elimina Duplicados
print(df.drop_duplicates(subset= 'review',inplace=True))
print()

#3.3 ELIMINACION DEL BR (CASO ESPECIAL NLP)
df['br_count'] = df['review'].str.count(r'<br\s*/?>', flags=re.IGNORECASE)
print(df['br_count'].value_counts().sort_index()) #Cuantas filas tienen 0,1,2,3
print()
print(df[df['br_count'] > 0]['review'].head())
print()
print("Total de <br /> en todo el dataset:", df['br_count'].sum())
print()
df['review'] = df['review'].str.replace(r'<br\s*/?>', ' ', regex=True)
print(df['review'])

#comprobacion de como quedaron las columnas 
print(df['sentiment'].value_counts()) #Muestra cuantos hay por cada columna

''' 
#INICIO DEL PASO EXTRA POR SER NLP
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

#ver palabras mas Importantes
print(vectorizer.get_feature_names_out()[:20])
print()

#FIN DEL PASO EXTRA POR SER NLP 

#Division de conjuntos
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

'''
