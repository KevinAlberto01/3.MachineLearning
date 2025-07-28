import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

df = pd.read_csv("1.Basic/Steps/3.1DataProcessing/IMDB Dataset.csv")

print()
print(df.shape)
print()
print(df.head(10))
print()
print(df.shape)
print()
print(df.info())
print()
print(df.describe())
print()
print(df['sentiment'].unique())
print()
print(df['sentiment'].value_counts())
print()
print(df.dtypes)
print()
df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})
print(df['sentiment'])
print()
print(df.dtypes)
print()
print(df.drop_duplicates(subset= 'review',inplace=True))
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
#Division de conjuntos
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

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

#Palabra mas frecuentes por clases
positive_text = ' '.join(df[df['sentiment'] == 1]['review'])
negative_text = ' '.join(df[df['sentiment'] == 0]['review'])

wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
wordcloud_neg = WordCloud(width=800, height=400, background_color='black').generate(negative_text)

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