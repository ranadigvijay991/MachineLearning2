# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:25:10 2021

@author:    Digvijay
"""
#%%
#Importing Libraries
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV,KFold,cross_val_score
from sklearn.linear_model import Ridge

#%%
#make Regression to make dataset
x,y = make_regression(n_samples=1000, n_features=4, shuffle = True,random_state=30)

print(x.shape)
print(y.shape)

#%%
#Initialization

trials = 30
score = 'r2'
tuned_parameters = [{'solver' : ['svd', 'lsqr'],'fit_intercept': ['True'],'normalize': ['False']},
                    {'solver' : ['sag', 'cholesky'],'fit_intercept': ['False'],'normalize': ['true']}]
non_nested_scores = np.zeros(trials)
nested_scores = np.zeros(trials)

#%%
#Iterating
for i in range(trials):
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    model= GridSearchCV(estimator = Ridge(), param_grid = tuned_parameters, scoring = score)
    model.fit(x, y)
    non_nested_scores[i] = model.best_score_
    
    
    # Nested CV with parameter optimization
    model = GridSearchCV(estimator= Ridge(), param_grid = tuned_parameters, cv=inner_cv, scoring= score)
    nested_score = cross_val_score(model, X=x, y=y, cv=outer_cv)
    nested_scores[i] = nested_score.mean()
#%%
#Score Difference
score_difference = non_nested_scores - nested_scores
#print(model.best_params_)
print("Average difference of {:6f} with std. dev. of {:6f}."
      .format(score_difference.mean(), score_difference.std()))
