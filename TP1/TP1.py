# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 10:31:59 2024

@author: lacou
"""

"""
import pandas as pd
from sklearn import datasets
import numpy as np
import matplotlib as mp
import seaborn as sns

#1
wine = datasets.load_wine()
#2
print(wine.data.shape)
print(wine.feature_names)
print(wine.target_names)
#3
df = pd.DataFrame(wine.data, columns =wine.feature_names)
df['target']=wine.target

#4
print(df.head()) #en argument le nombre de colomns

#5
print(df.info()) #affiche nom de colomne et nombre de ligne (peut observer si données manquantes)
print(df.dtypes)

#6
print(df.isnull().sum())

#7
df.iloc[:,0:3].describe()
# OU df[['alcohol', 'malic_acid, 'ash']].describe()

#•8*
freq=df['target'].value_counts()
freq.plot(kind='bar')

#9
df[['alcohol','magnesium', 'color_intensity']].hist()

#10
sns.pairplot(data=df,x_vars=['alcohol'],y_vars=['color_intensity'],hue='target',palette=['Red','Blue','Green'])

#res = pd.plotting.scatter_matrix(df,figsize=[12,12], c='y') 
"""

from sklearn import datasets
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#1-Chargement du dataset
wine = datasets.load_wine()

#2-Nombre d'échantillon et nombre attributs
print(wine.data.shape)
#Liste des attributs
print(wine.feature_names)
#Variable cible
print(wine.target_names)

#3-Convertir le dataset en DataFrame
df = pd.DataFrame(wine.data, columns = wine.feature_names)
df['target']=wine.target

#4-Afficher les 5 premières lignes
print(df.head())

#5-Afficher les types de données
print(df.dtypes)
print(df.info())

#6-Vérifier s’il y a des données manquantes
#print(df.info())
print(df.isnull().sum())

#7-Afficher une description statistique des attributs suivants : alcohol, malic_acid et ash
print(df[['alcohol','malic_acid','ash']].describe())
#OU df.iloc[:,0:3].describe()

#8-Afficher un graphique pour visualiser la fréquence de chaque classe.
freq= df['target'].value_counts()
freq.plot(kind='bar')
plt.figure(figsize=(12,10))

#9-Afficher un graphique pour visualiser la distribution des attributs suivants : alcohol, magnesium, color_intensity
df[['alcohol','magnesium','color_intensity']].hist()
#hist de distribution de tous les attributs
df.hist(figsize=(12,10))
#df['alcohol'].hist(bins=15)

#10-Afficher un graphique pour visualiser les rapports entre l’alcohol et color_intensity et les rapports entre l’alcohol et hue. Que remarquez-vous ?
sns.pairplot(data = df, x_vars=['alcohol'], y_vars=['color_intensity'],hue = 'target', palette = ['Red', 'Blue', 'limegreen'], diag_kind = None) 
plt.show()

#ou
"""
scatter= plt.scatter(df['alcohol'],df['color_intensity'])
plt.title('alcohol vs color_intensity')
plt.xlabel('alcohol')
plt.ylabel('color_intensity') #m : slope et b : intercept
m, b = np.polyfit(df['alcohol'], df['color_intensity'], 1) #1 : degre color_intensity = a *+ b* alcohol
plt.plot(df['alcohol'], m*df['alcohol'] + b, color='red')
plt.show()
"""

#12-KNN par défaut + performances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X=df.iloc[:,0:13] #OU X=df.drop("target",axis=1)
Y=df['target'].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=42 , test_size=0.2)
knn=KNeighborsClassifier()
knn.fit(X_train, Y_train)
print(knn.score(X_train,Y_train))
Y_pred = knn.predict(X_test)
print(knn.score(X_test,Y_test))
print(knn.accuracy_score(Y_pred,Y_test)) #taux de calssification, voir si le model est bien -> proche de 1

#13-KNN avec différentes valeurs de k 

for i in range(1,11):
        knn=KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, Y_train)
        Y_pred= knn.predict(X_test)
        print("LA valeur de k est :", i)
        print("k=3 ent",knn.score(X_train,Y_train))
        print("k=3 test",knn.score(X_test,Y_test),"\n")
        print(knn.accuracy_score(Y_pred,Y_test))
