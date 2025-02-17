#LIBRARYS#
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np 
from skimage.transform import resize 

##DATA SETS## 
digits = load_digits()
x = digits.data #Images 8x8 (plane format)
y = digits.target #(0-9)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)#Transform & Fit 
x_test = scaler.transform(x_test)#Transform the data 

###TRAIN MODEL###  
model = LogisticRegression(max_iter = 10000) #Start the model 
model.fit(x_train, y_train) #Train the model 

####EVALUATE THE MODEL####
score = model.score(x_test, y_test)
print(f'Accuracy: {score * 100:.2f}%')

#####PREDICT#####
y_pred = model.predict(x_test)

for i in range(10): #Show the prediction with the true value
    print(f"Prediction: {y_pred[i]}, True: {y_test[i]}")

######PREPARE THE VIZUALIZE RESULTS######
num_images = 10 #Numbers of images to show
fig, axes = plt.subplots(2, 5, figsize=(10,5)) #Figure with subsplots (2 rows, 5 columns)
axes = axes.ravel() #Flattens the array (iterate)

###VISUALIZE THE RESULTS#######
#"First Figure"#
for i in range(num_images):
    axes[i].imshow(x_train[i].reshape(8,8), cmap="gray")#Show original (8x8)
    axes[i].set_title(f"True: {y_test[i]}") #Title with true labels
    axes[i].axis('off') #Disable the axis

plt.tight_layout() #Adjust the layout to avoid overlap 


fig, axes = plt.subplots(2,5, figsize=(12, 5))
axes = axes.ravel() #Flatten the array (2D to 1D)

#"Second Figure"#
for i in range(num_images):
    img_resized = resize(x_train[i].reshape(8,8), (28,28), anti_aliasing = True) #Resize the image 
    axes[i].imshow(img_resized, cmap = "gray") #Reshape 8 x 8 
    axes[i].set_title(f"Pred: {y_pred[i]}, True: {y_test[i]}") #Title with labels (Predicted and True)
    axes[i].axis('off') #Disable the axis 

plt.tight_layout() #adjust the layout (not overlap)
plt.show() #Show Plot


