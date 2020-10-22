# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 

filename='ConjuntosDatos/tic-tac-toe.data'
data_tic = pd.read_csv(filename)

column_names=list(data_tic.columns)

datos=data_tic.to_numpy()

le = preprocessing.LabelEncoder()
datos_labels=le.fit_transform(datos)
X=datos_labels[:,0:-1]
y=datos_labels[:,-1]

'''
le = preprocessing.LabelEncoder()

clases_labels=le.fit_transform(clases_dataset)
labels=le.fit_transform(clases_dataset)

for i in range(atribute_dataset.shape[1]):
    labels.append(le.fit_transform(atribute_dataset[:,i]))
'''    
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 

print(X_test)

gnb = GaussianNB().fit(clases_labels, labels) 
gnb_predictions = gnb.predict(X_test) 