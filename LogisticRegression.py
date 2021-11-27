#%%
#importing libraries
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#%%
#making dataset
X,y = make_classification(n_samples=1000, n_features=1, n_informative=1,
        n_redundant=0, n_repeated=0,n_classes=2,n_clusters_per_class=1)
print(X)
print(y)
#%%
print(X.shape,y.shape)
#%%
#Visualization
plt.scatter(X,y)

#%%
#test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
y_test = np.reshape(y_test,(200,1))

#%%
#model fitting
model = LogisticRegression()
model.fit(X_train,y_train)

#%%
#Predicting
y_pred = model.predict(X_test)
print(y_pred)

#%%
#Accuracy Score
acc = accuracy_score(y_test,y_pred)
print(acc)

#%%
#ACCURACY and CONFUSION MATRIX
print(f"ACCURACY : {acc*100}%")
print("CONFUSION MATRIX : \n",confusion_matrix(y_test, y_pred))

#%%
w = [0.0, 0.0]
w = np.reshape(w,(2,1))
y_p = np.dot(X,w)
y_p = 1/(1+np.exp(y_p))
a = (y_train*np.log(y_p)) + ((1-y)*np.log(1-y_p))
print(a)
#%%
#Cost Function
y_p = ((X_test[:,0] * model.coef_[: , 0]) + (X_test[:,1] * model.coef_[: , 1]) + (X_test[:,2] * model.coef_[: , 2])
       + (X_test[:,3] * model.coef_[: , 3])) + model.intercept_

y_p = 1 / (1 + np.exp(-y_p))


m = y_test.shape[0]

c = -(1/m) * np.sum((y_test * np.log(y_p)) + ((1 - y_test) * np.log(1 - y_p)))

print('cost : ',c)
