# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 18:36:33 2020

@author: quaxoc
"""

from datos import *
from test_train import *
import pandas as pd
import numpy as np

dataset=Datos('ConjuntosDatos/tic-tac-toe.data')
#dataset=Datos('ConjuntosDatos/german.data')

rows_number=dataset.datos.shape[0]
test_proportion=0.3
line_ids=validacion_cruzada(rows_number,4)
line_ids_test=line_ids[0]['Test']
line_ids_train=line_ids[0]['Train']


#print("Test",line_ids_test)
#print("Train",line_ids_train)
#print(dataset.extraeDatos(line_ids_test))

print(dataset.nominalAtributos)
print(dataset.diccionario)
class_name="Class"
atr_val=["x","x","x","o","b","x","b","o","o"]
p_tables=[]
print(atr_val)
'''
for atr, atr_values in dataset1.diccionario.items():
    if atr!= class_name:
        print(atr)
        p_atr=[]
        #Creating p table 
        for class_type in dataset1.diccionario[class_name]:
            c=[]
            for at in atr_values:
                c.append(at)
            print(c)    
            np.c_[p_atr,c]
 '''    

laplace=True
train=dataset.extraeDatos(line_ids_test)
#train=dataset.datos
train_values=train[:,0:-1]
train_classes=train[:, -1]


classes=dataset.diccionario[-1]
atr=dataset.diccionario[:-1]

#print(classes)
p_priori=[]
for c in classes:
    p_priori.append(np.sum(train_classes==c)/len(train_classes))
print(p_priori)

#Calculando tablas de probabilidad dada la clase, clases en columnas, atributos en filas
prob_dada_clase=[]
a_id=0
for a in atr:
    if dataset.nominalAtributos[a_id]:
        p_table=np.empty([len(a), len(classes)])
        i=0
        for c in classes:
            j=0
            for aval in a.keys():
                #a_occur=np.sum(c_data==aval)
                #print(train[np.where((train[:,-1]==c) & (train[:,i]==aval))].shape[0])
                p_table[j,i]=train[np.where((train[:,-1]==c) & (train[:,a_id]==aval))].shape[0]
                j+=1
            i+=1
        #aplicando Laplace
        if (0 in p_table) and laplace:
            p_table=p_table+1
        #Convirtiendo occuriencias en probabilidades
        p_table/=np.sum(p_table, axis=0)
        prob_dada_clase.append(p_table)
    a_id+=1
        
print(prob_dada_clase)


#Probabilidad clase:
#P(C|A1..An)=P(A1|C)..P(An|C)P(C)/P(A1).. P(An)
#P(C1|..)=P(C1|..)/Sum(P(Ci|A1.An))

ids=[2,2,2,1,0,2,0,1,1]
P_posteriori=[]
for c_i in range(len(classes)):
    j=0
    p_c_post=p_priori[c_i]
    for atr_val in ids:
        p_c_post=p_c_post*prob_dada_clase[j][atr_val,c_i]
        j+=1
    P_posteriori.append(p_c_post)
 
P_posteriori=P_posteriori/np.sum(P_posteriori)    
print(P_posteriori)
index_max = np.argmax(P_posteriori)

#Return predicted class 
print(list(classes.keys())[list(classes.values()).index(index_max)])