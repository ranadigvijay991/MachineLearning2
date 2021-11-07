#Importing Libraries
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

#%%
#Dataset
x,y = make_regression(n_samples = 100 , n_features = 5)
print(x.shape)
print(y.shape)


#%%
#Polynomial Features
poly_X = PolynomialFeatures(3)
poly_features = poly_X.fit_transform(x)
print(poly_features.shape)

#%%
#Plot
plt.scatter(poly_features[:,1] , y,marker=('.'))
plt.scatter(poly_features[:,1], y,marker=('.'))
plt.scatter(poly_features[:,2],y,marker=('.'))
plt.scatter(poly_features[:,3],y,marker=('.'))
plt.scatter(poly_features[:,4],y,marker=('.'))

plt.show()

#%%
#Test train split
X_train,X_test,y_train,y_test = train_test_split(poly_features , y , test_size=0.2)

print('Shape of X_train : ' , X_train.shape)
print('Shape of y_train : ' , y_train.shape)
print('Shape of X_test : ' , X_test.shape)
print('Shape of y_test : ' , y_test.shape)

#%%
#Model training and prediction
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

print('accuracy : ' , reg.score(X_test , y_test))   

#%%
#RIDGE
from sklearn.linear_model import Ridge
clf = Ridge()
clf.fit(X_train,y_train)
y_pred2 = reg.predict(X_test)

print('accuracy : ' , clf.score(X_test , y_test))   

#%%
from sklearn.linear_model import RidgeCV

model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])

model.fit(X_train , y_train)
y_pred3 = model.predict(X_test)

print('accuracy : ' , model.score(X_test , y_test))

#%%
#MSE
from sklearn.metrics import mean_squared_error
print('MSE : ' , mean_squared_error(y_test, y_pred))

#%%
#MLSE
from sklearn.metrics import mean_squared_log_error
print('MSLE : ' ,  mean_squared_log_error(np.absolute(y_test) , np.absolute(y_pred2)))

#%%
#R2 score
from sklearn.metrics import r2_score
print('R2 score : ' , r2_score(y_test , y_pred2)) 

#%%
plt.plot(X_train , reg.coef_* X_train + reg.intercept_ , '-b')
#%%
plt.plot(X_train , clf.coef_* X_train + clf.intercept_ , '-r')
#%%
plt.plot(X_train , model.coef_* X_train + model.intercept_ , '-y')













