# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:58:41 2020

@author: quaxoc
"""

import numpy as np
from sklearn import preprocessing
from datos import *
from sklearn.naive_bayes import GaussianNB

dataset=Datos('ConjuntosDatos/tic-tac-toe.data')
data_table=dataset.datos
le = preprocessing.LabelEncoder()

a=np.empty([data_table.shape[0], data_table.shape[1]])
for i in range(data_table.shape[1]):
    column=data_table[:,i]
    column=le.fit_transform(column)
    a[:,i]=np.transpose(column)

print(a)
input_array=[]
for row in range(a.shape[0]):
    #print (a[row,:])
    input_array.append(a[row,0:-1])
classes_encoded=a[:,-1]


clf = GaussianNB()

#clf = GaussianNB()
clf.fit(input_array, classes_encoded)
print(clf.score(input_array, classes_encoded))
print(a[10])
print(clf.predict([input_array[10]]))

confusion_matrix=np.zeros([2,2])
for r in range(len(classes_encoded)):
    predicted=clf.predict([input_array[r]])
    #print("Predicted ", predicted, "Real ", classes_encoded[r])
    if predicted==1:
        if classes_encoded[r]==1:
            confusion_matrix[0,0]+=1
        else: 
            confusion_matrix[0,1]+=1
    else:
        if classes_encoded[r]==0:
            confusion_matrix[1,1]+=1
        else: 
            confusion_matrix[1,0]+=1
            
print(confusion_matrix)