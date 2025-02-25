#1.DECLARATE LIBRARIES
import numpy as np 
#import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


#2.LOAD THE DATASET(8x8)
digits = load_digits()
x = digits.data #Each images ius 8x8(64)
y = digits.target #Target (0-9)

#3.EXPLORING THE DIMENSIONS OF THE DATASET
print(f"Dimension of x: {x.shape}") #(1797, 64) "1797 images with 64 pixels"
print(f"Dimension of y: {y.shape}") #(1797) "1797 labels"

#4.CHECK THE CLASSES AND THE NUMBER OF EXAMPLES PER CLASS
clases, count_classes = np.unique(y,return_counts = True)
print(f"Classes: {clases}")
print(f"Number of examples per class: {count_classes}")

#5.SPLIT INTO TRAIN AND TEST SETS
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


#6.NORMALIZATION WITH STANDARS SCALER
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("Datas normalized (first 5 rows):")
print(x_train[:5])

#7. DEFINE THE MODEL 
model =  SVC(kernel = 'linear')

#8.TRAIN MODELS AND EVALUATE 
model.fit(x_train, y_train)

#9.EVALUATION METRICS

#9.1 Predictions
y_pred = model.predict(x_test)
#9.2 Accuracy
accuracy = model.score(x_test, y_test)
print(f"Accuracy: {accuracy: .4f}")
#9.3 Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
#9.4 Classification Report
classification_reports = classification_report(y_test, y_pred)

#10.VISUALIZATION
#10.1 Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues',
                xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
#10.2 Print aditional INformation
print("Confusion Matrix:")
print(conf_matrix)

#10.3 Find the most misclassified digits
errors = np.where(conf_matrix != np.diag(conf_matrix))
print("\nMost misclassified digits:") 
for true_label, pred_label in zip(errors[0], errors[1]):
    print(f" - The model confused {true_label} with {pred_label} {conf_matrix[true_label]} times.")
print("-" * 40)

#10.4 Classification Report
print("Classification Report:")
print(classification_reports)
print("-" * 50)

print("Evaluation Completed!")

#11.OPTIMIZATION (TUNING & HYPERPARAMETERS)