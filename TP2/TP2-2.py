# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:48:17 2024

@author: lacou
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#1-Charger le fichier csv
df= pd.read_csv('petrol_consumption.csv')

#2-explorer la dataframe
print(df.info())
print(df.describe())
print(df.isnull().sum())

#3-ensemble de données d'entrainement et de test
X = df.drop("Petrol_Consumption",axis=1).values
y=df["Petrol_Consumption"].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#4-Entrainer un modèle de regression linéaire avec l'ensemble d'entrainement
reg=LinearRegression()
reg.fit(X_train,y_train.reshape(-1,1))

#5-Prédire les valeurs de l'ensemble de test



