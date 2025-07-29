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

#2.INSPECCION INICIAL
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

#3.LIMPIEZA DE DATOS

#3.2 Elimina Duplicados
print(df.drop_duplicates(subset= 'review',inplace=True))
print()

#3.3 Eliminacion de br (para NLP)
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
#print(df['sentiment'].value_counts()) #Muestra cuantos hay por cada columna

#APARTIR DE AQUI FALTA DOCUMENTAR !!!!! IMPORTANTE
# 3.4 LIMPIEZA DE TEXTO ADICIONAL
def limpiar_texto(texto):
    texto = texto.lower()  # minúsculas
    texto = re.sub(r'<.*?>', ' ', texto)  # etiquetas HTML restantes
    texto = re.sub(r'\d+', '', texto)  # eliminar números
    texto = texto.translate(str.maketrans('', '', string.punctuation))  # eliminar signos puntuación
    texto = re.sub(r'\s+', ' ', texto)  # eliminar espacios múltiples
    return texto.strip()

df['review_clean'] = df['review'].apply(limpiar_texto)


#INICIO PARA PARTE ESPECIAL PARA WORDCLOULD
# 3.5 Eliminar stopwords personalizadas #
# Stopwords en inglés + personalizadas
stop_words = list(text.ENGLISH_STOP_WORDS.union(stopwords.words('english')))
stop_words += ['movie', 'film', 'one', 'character', 'time', 'story', 'make', 'see',
    'scene', 'way', 'thing', 'look', 'plot', 'work', 'director', 'watch',
    'get', 'go', 'going', 'even', 'bit', 'really', 'know', 'think',
    'much', 'well', 'take', 'still', 'say', 'something', 'lot', 'back',
    'also', 'end', 'though', 'better', 'people', 'little', 'nothing',
    'makes', 'right', 'man', 'woman', 'new', 'life', 'im'
]

df = df.dropna(subset=['review_clean'])
# Vectorizador actualizado
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=5000)
#FINAL PARA PARTE ESPECIAL PARA WORDCLOULD

#5.TRANSFORMACION DE VARIABLES CATEGORICAS (PARA NLP)
#df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0}) ESTO SE CAMBIARA PARA ABAJO
#print(df['sentiment'])
#print()
#print(df.dtypes)
#print()
#Crear el vectorizador
#Aplicar vectorizacion al texto 
x = vectorizer.fit_transform(df['review_clean'])
#Etiquetas
y = df['sentiment']
print("Form of X:", x.shape)
print()

#ver palabras mas Importantes
print(vectorizer.get_feature_names_out()[:20])
print()

#FIN DEL PASO EXTRA POR SER NLP 

#7.DIVISION DE CONJUNTOS 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

################# A.FIN DE DATA PROCESSING #################

############### B.EXPLORATORY DATA ANALISYS ################
#comparativa 
# Comparativa del punto 3.4
# Longitud ANTES de limpiar
df['length_original'] = df['review'].apply(lambda x: len(x.split()))
# Longitud DESPUÉS de limpiar
df['length_clean'] = df['review_clean'].apply(lambda x: len(x.split()))

sns.histplot(df['length_original'], color='blue', label='Original', bins=50, alpha=0.5)
sns.histplot(df['length_clean'], color='green', label='Cleaned', bins=50, alpha=0.5)
plt.legend()
plt.title('Distribución de longitud de reseñas (antes vs después)')
plt.xlabel('Número de palabras')
plt.show()

#PASO 2
#Distribucion de clases
sns.countplot(x='sentiment', data= df)
plt.title('Distribucion de clases (0=Negativo), 1 = Poositivo)')
plt.xticks([0,1], ['Negative', 'Positive'])
plt.show()

#Longitud de las reseñas
df['review_length'] = df['review'].apply(lambda x: len(x.split()))
sns.histplot(data= df, x='review_length', hue='sentiment', bins=50)
plt.title('Distribucion de longitud de reseñas')
plt.xlabel('Numero de palabras')
plt.ylabel('Frecuencia')
plt.show()


#Palabra mas frecuentes por clasesreview_clean
# Convertir a set para más velocidad en filtrado

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

#5.TRANSFORMACION DE VARIABLES CATEGORICAS (PARA NLP)
#df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0}) #ESTO ESTABA ARRIBA 
########### B.FIN DE EXPLORATORY DATA ANALISYS #############