import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

df = pd.read_csv("1.Basic/Steps/3.1DataProcessing/IMDB Dataset.csv")

print()
#Inspeccion de tamaño
'''print(df.shape)
print()
print(df.info())
print()
print(df.describe())
print()
print(df.head(10))
print()
print(df['sentiment'].unique())
print()
print(df['sentiment'].value_counts())
print()
print(df.dtypes)
'''


#3.Limpieza de datos 
print(df.drop_duplicates(subset= 'review',inplace=True))

print()

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

#FIN DEL PASO EXTRA POR SER NLP 


#ver palabras mas Importantes
print(vectorizer.get_feature_names_out()[:20])
print()
#Division de conjuntos
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)
