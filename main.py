# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 18:36:33 2020

@author: quaxoc
"""

from datos import *
from test_train import *
import pandas as pd
import numpy as np
from scipy.stats import norm

'''
Input:
    dataset - needed to read the dataset atributes
    train - train data obtained by test_train
    test_line - the line with attribute values for the class prediction
    laplace - True or False depending if Laplace is used for categorical Naive Bayes
'''
def naive_bayes(dataset, train, test_line, laplace):
    #The list of classes with ids
    classes=dataset.diccionario[-1]
    #The list of atributes with ids
    atr=dataset.diccionario[:-1]


    #Convert test line to ids using the dictionary
    test_line_ids=[]
    for k in range(len(test_line)):
        if dataset.nominalAtributos[k]:
            test_line_ids.append((atr[k][test_line[k]]))
        else:
            test_line_ids.append(test_line[k])
     
      
    train_classes=train[:, -1]
    p_priori=[]
    for c in classes:
        p_priori.append(np.sum(train_classes==c)/len(train_classes))
    #print("Prior probabilities of classes:", p_priori)
    
    #An array with tables for each attribute that either contains probabilities or mean and std for non categorical values
    prob_dada_clase=[]
    a_id=0
    for a in atr:
        if dataset.nominalAtributos[a_id]:
            p_table=np.empty([len(a), len(classes)])  #Creating a table that has probabilities for each atribute (rows) considering the class (columns)
            i=0
            for c in classes:
                j=0
                for aval in a.keys():
                    p_table[j,i]=train[np.where((train[:,-1]==c) & (train[:,a_id]==aval))].shape[0]
                    j+=1
                i+=1
            #aplicando Laplace
            if (0 in p_table) and laplace:
                p_table=p_table+1
            #Convirtiendo occuriencias en probabilidades
            p_table/=np.sum(p_table, axis=0)
        else:   #Apply gaussian bayes
            p_table=np.empty([2, len(classes)])  #Each column of the table is a class, first row is mean and second is standard deviation of the data values
            i=0
            for c in classes:
                p_table[0,i]=np.mean(train[np.where(train[:,-1]==c)][:, a_id])
                p_table[1,i]=np.std(train[np.where(train[:,-1]==c)][:, a_id])
                i+=1
        prob_dada_clase.append(p_table)
        a_id+=1
            
    
    #Probabilidad clase:
    #P(C|A1..An)=P(A1|C)..P(An|C)P(C)/P(A1).. P(An)
    #P(C1|..)=P(C1|..)/Sum(P(Ci|A1.An))
    
    P_posteriori=[]
    for c_i in range(len(classes)):
        j=0
        p_c_post=p_priori[c_i]
        for atr_val in test_line_ids:
            if dataset.nominalAtributos[j]:
                p_c_post=p_c_post*prob_dada_clase[j][atr_val,c_i]
            else:
                mu=prob_dada_clase[j][0,c_i]
                std=prob_dada_clase[j][1,c_i]
                p_contin=norm.pdf(atr_val, mu, std)
                p_c_post=p_c_post*p_contin
                print("Val: ", atr_val, "Mu ", mu, "Std ", std, "P(val)", p_contin)
            j+=1
        P_posteriori.append(p_c_post)
     
    P_posteriori=P_posteriori/np.sum(P_posteriori)    
    index_max = np.argmax(P_posteriori)
    
    #Return predicted class 
    predicted_class=list(classes.keys())[list(classes.values()).index(index_max)]
    return(predicted_class, P_posteriori)    

    
    
    
    
    
    
    
    
    
    
    
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

#print(dataset.nominalAtributos)
#print(dataset.diccionario)

#atr_val=["x","x","x","o","b","x","b","o","o"]

#print(atr_val)



    
laplace=True
#test_line=["A14",24,"A32","A43",3430,"A63","A75",3,"A93","A101",2,"A123",31,"A143","A152",1,"A173",2,"A192","A201"]
test_line=["x","x","x","o","b","x","b","o","o"]

train=dataset.extraeDatos(line_ids_test)

class_name, P_classes = naive_bayes(dataset, train, test_line, laplace)
print("Predicted class: ", class_name, "Probabilities for classes: ", P_classes)

