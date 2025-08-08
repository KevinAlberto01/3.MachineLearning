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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

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
#ENTRENAMIENTO BASICO 
'''
modelL = LogisticRegression()
modelL.fit(X_train, y_train)
predictions = modelL.predict(X_test)
'''
#5.HIPERPARAMETERS
# Lista de valores de C a probar (regulación inversa)
'''
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

for c in C_values:
    modelL = LogisticRegression(C=c, solver='liblinear', random_state=42)
    modelL.fit(X_train, y_train)
    predictions = modelL.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"C={c} --> Accuracy: {acc:.4f}")
print()
'''
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

#======================== B.INICIO DE "K-Nearest Neighbors (KNN)" =======================
#ENTRENAMIENTO BASICO 
'''
modelK = KNeighborsClassifier(n_neighbors=5)
modelK.fit(X_train, y_train)
predictions = modelK.predict(X_test)
print("Accuracy:", accuracy_score(y_test,predictions))
'''
#5.HIPERPARAMETERS
'''
from sklearn.model_selection import cross_val_score
for k in range(11, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"k={k}: Accuracy promedio = {scores.mean():.4f}")
'''
#5.1 After Optimization
'''
modelK = KNeighborsClassifier(n_neighbors=16)
modelK.fit(X_train, y_train)
predictions = modelK.predict(X_test)
print("Accuracy:", accuracy_score(y_test,predictions))
'''
#EVALUATION METRICS 
'''
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

y_prob = modelK.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test,y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label = f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
'''
#========================== B.FIN DE "K-Nearest Neighbors (KNN)" ========================

#============================== C.INICIO DE "DECISION TREE " ============================
#ENTRENAMIENTO BASICO 
'''
modelDT = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=42)
modelDT.fit(X_train, y_train)
predictions = modelDT.predict(X_test)
print("Accuracy:", accuracy_score(y_test,predictions))
'''
#5.HIPERPARAMETERS
'''
for d in range(3, 21, 2):
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"max_depth={d} --> Accuracy: {acc:.4f}")
'''
#5.1 After Optimization
'''
modelDT = DecisionTreeClassifier(criterion='gini', max_depth=19, random_state=42)
modelDT.fit(X_train, y_train)
predictions = modelDT.predict(X_test)
print("Accuracy:", accuracy_score(y_test,predictions))
'''
#EVALUATION METRICS 
'''
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

y_prob = modelDT.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test,y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label = f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
'''
#================================ C.FIN DE "DECISION TREE " ============================

#2.FIN DE LOS ALGORITMOS BASICOS

#===================== D.INICIO DE "SUPPORT VECTOR MACHINE (SVM)" =======================
#ENTRENAMIENTO BASICO 
'''
model = make_pipeline(StandardScaler(with_mean=False), SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,predictions))
'''
#5.HIPERPARAMETERS
'''
C_values = [0.1,1,10]
gamma_values = [scale,0.01,0.1,1]

for C in C_values:
    for gamma in gamma_values:
        model = make_pipeline(
            StandardScaler(with_mean=False),
            SVC(kernel='rbf', C=C, gamma=gamma, probability=True)
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"C={C}, gamma={gamma} --> Accuracy: {acc:.4f}")
'''
#5.1 After Optimization
'''
model = make_pipeline(StandardScaler(with_mean=False), SVC(kernel='rbf', C=10, gamma='scale', probability=True))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,predictions))
'''
#EVALUATION METRICS 
'''
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

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test,y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label = f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
'''

#======================= D.FIN DE "SUPPORT VECTOR MACHINE (SVM)" ========================

#============================= E.INICIO DE "RANDOM FOREST" ==============================
#ENTRENAMIENTO BASICO 
'''
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,predictions))
'''
#5.HIPERPARAMETERS
'''
for n in [50, 100, 150]:
    for d in range(3, 21, 2):
        model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"n_estimators={n}, max_depth={d} --> Accuracy: {acc:.4f}")
'''
#5.1 After Optimization
'''
model = RandomForestClassifier(n_estimators=150, max_depth=19, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,predictions))
'''
#EVALUATION METRICS 
'''
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

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test,y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label = f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
'''
#=============================== E.FIN DE "RANDOM FOREST" ===============================
#============================ F.INICIO DE "NEURAL NETWORKS" =============================
#ENTRENAMIENTO BASICO 
'''
model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=300, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,predictions))
'''
#5.HIPERPARAMETERS
'''

layer_configs = [
    (50,),
    (100,),
    (100, 50),
    (150, 100, 50),
    (200, 100, 50)
]

for config in layer_configs:
    model = MLPClassifier(hidden_layer_sizes=config, activation='relu', max_iter=300, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"hidden_layer_sizes={config} --> Accuracy: {acc:.4f}")
'''    
#5.1 After Optimization
'''
model = MLPClassifier(hidden_layer_sizes=(200, 100, 50), activation='relu', max_iter=300, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,predictions))
'''
#EVALUATION METRICS 
'''
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

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test,y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label = f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
'''
#============================= F.FIN DE "NEURAL NETWORKS" ===============================
