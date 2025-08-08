#0.LIBRERIA 
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
#LIBRERIAS PARA ALGORITMOS
from sklearn.linear_model import LogisticRegression

#LIBRERIAS PARA METRICS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,confusion_matrix,classification_report
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import roc_curve, roc_auc_score

#EXPORT MODEL
import joblib
from sklearn.pipeline import Pipeline

#################### A.DATA PROCESSING ####################

#1.CARGA DE DATOS
df = pd.read_csv("1.Basic/Steps/3.1DataProcessing/IMDB Dataset.csv")
print()
print("1.SE LEYO LA BASE DE DATOS CORRECTAMENTE")
print("La base de datos cuenta con un tamaño de:",df.shape)
print()

#3.LIMPIEZA DE DATOS

#3.2 Elimina Duplicados
df.drop_duplicates(subset= 'review',inplace=True)
print("2.SE ELIMINO LOS DATOS DUPLICADOS")
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
    return texto.strip()

df['review_clean'] = df['review'].apply(limpiar_texto) #aqui se agrega una nueva columna pero es solamente la correccion
print("La base de datos tiene un tamaño de:",df.shape)

#REVISION DEL FORMATO ESTE NOMAS ES PARA CONFIRMAR POR SI VEO UNA ANOMALIA
#print(df['review_clean'].head(10))

################# A.FIN DE DATA PROCESSING #################

############### B.EXPLORATORY DATA ANALISYS ################
#Longitud de las reseñas
df['review_length'] = df['review'].apply(lambda x: len(x.split()))
sns.histplot(data= df, x='review_length', hue='sentiment', bins=50)
plt.title('Distribucion de longitud de reseñas')
plt.xlabel('Numero de palabras')
plt.ylabel('Frecuencia')
plt.show()
print()
print("5.SE GENERO LA GRAFICA DE LONGITUD DE LAS RESEÑAS")


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
#Aplicar vectorizacion al texto 

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
print()
print("6.SE GENERO LA GRAFICA DE WORDCLOUD")


############ B.FIN DE EXPLORATORY DATA ANALISYS #############


######### C.INICIO DE TRAINING MULTIPLE ALGORITHMS ##########

#5.TRANSFORMACION DE VARIABLES CATEGORICAS (PARA NLP)
df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})
print(df['sentiment'])

#1.INICIO PREPARACION DE DATOS
x = vectorizer.fit_transform(df['review'])
y = df['sentiment']
#Etiquetas

#FIN DEL PASO EXTRA POR SER NLP 

#7.DIVISION DE CONJUNTOS 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print()
print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)
print()
#1.FIN DE PREPARACION DE DATOS

#2.INICIO DE LOS ALGORITMOS BASICOS

#========================== A.INICIO DE "LOGISTIC REGRESSION" ===========================
#5.1 AFTER TUNINIG
# Entrenamos el modelo final con el mejor valor encontrado (C=1)

modelL = LogisticRegression(C=1, solver='liblinear', random_state=42)
modelL.fit(X_train, y_train)
predictions = modelL.predict(X_test)

#EVALUATION METRICS -basic
print("Accuracy:", accuracy_score(y_test,predictions))

#EVALUATION METRICS 
print("Accuracy:", accuracy_score(y_test,predictions))
print("Precision:", precision_score(y_test,predictions))
print("Recall:", recall_score(y_test,predictions))
print("F1 Score:", f1_score(y_test,predictions))
print("Confusion Matrix: \n", confusion_matrix(y_test,predictions))
#general summary
print("\nClassification Report:\n", classification_report(y_test,predictions))
print()

disp=ConfusionMatrixDisplay.from_predictions(y_test,predictions, cmap='Blues')
plt.show()

y_prob = modelL.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test,y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label = f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

#============================ A.FIN DE "LOGISTIC REGRESSION" ============================
######### C.FIN DE TRAINING MULTIPLE ALGORITHMS ##########

#D.EXPORTACION DEL MODELO
# División con texto crudo
X_train, X_test, y_train, y_test = train_test_split(df['review_clean'], df['sentiment'], test_size=0.2, random_state=42)

# Pipeline con vectorizador configurado igual
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stop_words, max_features=5000)),
    ('model', LogisticRegression(C=1, solver='liblinear', random_state=42))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, '1.Basic/Steps/3.6JoinAll(1-2)/LogisticPipeline.pkl')