# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:23:16 2024

@author: lacou
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#1-Charger le fichier csv
score= pd.read_csv('student_scores.csv')  #OU avec chemin r'C:\Users\lacou\Documents\ESME\4-Ingé 2\Machine learning\TP2'

#2-explorer la dataframe
print(score.info())
print(score.describe())
print(score.isnull().sum())

#3-Entrainer un modèle de regression linéaire sur l'ensemble des données
X1=score['Hours'].values
X2=score['Hours']
X=score['Hours'].values.reshape(-1,1) 
y=score['Scores'].values.reshape(-1,1)
reg=LinearRegression()
reg.fit(X,y)

#4-Scores=a*Hours+b Calculer a et b
a=(score["Scores"][24]-score["Scores"][0])/(score["Hours"][24]-score["Hours"][0])
b=score["Scores"][0]-a*score["Hours"][0]

print("a: ", reg.coef_)
print("b: ", reg.intercept_)


#5-Tracer le nuage de points avec plt.scatter()
plt.scatter(X,y)
plt.ylabel('Scores')
plt.xlabel('Hours')
plt.title("Scores vs Hours")
plt.show()

#6-Superposer le tracé du nuage de points avec la ligne de regression linéaire avec plt.plot()
y_pred=reg.coef_*X+reg.intercept_
plt.scatter(X,y)
plt.ylabel('Scores')
plt.xlabel('Hours')
plt.plot(X,y_pred,color='r')





