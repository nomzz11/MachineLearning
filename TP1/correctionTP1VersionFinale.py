# -*- coding: utf-8 -*-
"""

@author: MAROUANE
"""

from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#1.charger le dataset
wine = load_wine()
#nbre échantillons et attributs
print(wine.data.shape)
#afficher la liste des attriburs et la variable cible
print (wine.feature_names)
print (wine.target_names)

#Q3. convertir en dataframe 
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target']=wine.target

#Q4. afficher les 5 premières lignes
print(df.head())
#Q5.afficher les types de données
#print(df.info())
print(df.dtypes)
#Q6. 
print(df.isnull().sum())

#Q8. 
df.iloc[:,0:3].describe()

df[['alcohol','malic_acid', 'ash']].describe()

#Q8. graphique pour visualiser la fréuence de chaque classe
freq= df['target'].value_counts()
freq.plot(kind='bar')


plt.figure(figsize=(12,10))


#Q9. hist de distribution des attributs : alcohol','magnesium','color_intensity'
df[['alcohol','magnesium','color_intensity']].hist()
#hist de distribution de tous les attributs
df.hist(figsize=(12,10))
#df['alcohol'].hist(bins=15)


#Q10. un graphique pour visualiser les rapports entre l’alcohol et color_intensity et les rapports entre l’alcohol et hue
#Q10. version 1
import seaborn as sns
sns.pairplot(data = df, x_vars=['alcohol'], y_vars=['color_intensity'],hue = 'target', palette = ['Red', 'Blue', 'limegreen'], diag_kind = None) 
plt.show()

sns.pairplot(data = df, x_vars=['alcohol'], y_vars=['hue'],hue = 'target', palette = ['Red', 'Blue', 'limegreen'], diag_kind = None) 
plt.show()

#Q10. version 2
scatter= plt.scatter(df['alcohol'],df['color_intensity'])
plt.title('alcohol vs color_intensity')
plt.xlabel('alcohol')
plt.ylabel('color_intensity') #m : slope et b : intercept
m, b = np.polyfit(df['alcohol'], df['color_intensity'], 1) #1 : degre color_intensity = a *+ b* alcohol
plt.plot(df['alcohol'], m*df['alcohol'] + b, color='red')
plt.show()

#Q11. affiche boite moustache pour les atttributs montre la dispersion indique la valeur médiane, min max et l'alcol n'a pas de valeur abberantes
df[['alcohol','magnesium','color_intensity']].plot.box()

#apprentissage KNN

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Q12
#features
X = df.iloc[:,0:13]
#X = df.drop('target',axis=1).values
#variable cible
y = df['target'].values
#données d'entrainement et données de test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
#instance de k plus proches voisins
knn1= KNeighborsClassifier()
#entrainer le modèle
knn1.fit(X_train, y_train)
#faire les prédictions
ypred = knn1.predict(X_test)
#précision des prédictions
print(accuracy_score(ypred,y_test))
#performance du modèle sur les données de test
print(knn1.score(X_test,y_test))
#performance du modèle sur les données d'entrainement
print(knn1.score(X_train,y_train))

#Q13
for i in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    print ("la valeur de k :", i)
    print(accuracy_score(y_test, y_pred))
    print(knn.score(X_test,y_test))
    print(knn.score(X_train,y_train))
    print("--------------------------------------------")