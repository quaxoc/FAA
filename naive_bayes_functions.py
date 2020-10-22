# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 18:36:33 2020

@author: quaxoc
"""

from datos import *
import pandas as pd
import numpy as np
import random 
from scipy.stats import norm

'''
Test - train module functions: split, validacion_simple, validacion_cruzada
'''
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def validacion_simple(row_count,test_proportion):
    id_list=list(range(row_count))
    random.shuffle(id_list)
    cut=round(test_proportion*row_count)
    sampling_test=sorted(id_list[0:cut])
    sampling_train = sorted(id_list[cut:])
    sampling={"Test":sampling_test,
              "Train":sampling_train}
    return(sampling)
    #print(sampling_test)
    #print(sampling_train)
    
def validacion_cruzada(row_count,k):
    
    id_list=list(range(row_count))
    n, m = divmod(row_count, k)
    
    random.shuffle(id_list)
    #print(id_list)
    sampling=[]
    chunks=[]
    for i in range(k):
        chunks.append(sorted(id_list[i*n+min(i,m):(i+1)*n+min(i+1,m)]))
    #print(chunks)
    
    for i in range(k):
        train=chunks.copy()
        test=train.pop(i)
        train1d=[]
        for ch in train:
            train1d+=ch
        dict_tt={"Test":test,
              "Train":sorted(train1d)}
        sampling.append(dict_tt)


        #sampling.append(validacion_simple(row_count,test_proportion))
    return(sampling)

'''
Naive bayes module
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
    
    
    #Entrenamiento
    
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
            
        
    #Clasifica 
    
    
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
            j+=1
        P_posteriori.append(p_c_post)
     
    P_posteriori=P_posteriori/np.sum(P_posteriori)    
    index_max = np.argmax(P_posteriori)
    
    #Return predicted class 
    predicted_class=list(classes.keys())[list(classes.values()).index(index_max)]
    return(predicted_class, P_posteriori)    

    
   
    
    
    
#dataset=Datos('ConjuntosDatos/tic-tac-toe.data')
dataset=Datos('ConjuntosDatos/german.data')
    
laplace=True


#Validación

#Validation part
rows_number=dataset.datos.shape[0]


#Simple validation
test_proportion=0.3
line_ids=validacion_simple(rows_number,test_proportion)
line_ids_test=line_ids['Test']
line_ids_train=line_ids['Train']

train=dataset.extraeDatos(line_ids_train)
test=dataset.extraeDatos(line_ids_test)

counter=0
for test_line in test:
    class_name, P_classes = naive_bayes(dataset, train, test_line[0:-1], laplace)
    if class_name == test_line[-1]:
        counter+=1
assert_simple=counter/len(test)
print("Simple validation assert:", assert_simple)


#Cross validation
partitions=4
line_ids=validacion_cruzada(rows_number,partitions)
assert_cross=[]
for i in range(partitions):
    line_ids_test=line_ids[i]['Test']
    line_ids_train=line_ids[i]['Train']
    train=dataset.extraeDatos(line_ids_train)
    test=dataset.extraeDatos(line_ids_test)
    counter=0
    for test_line in test:
        class_name, P_classes = naive_bayes(dataset, train, test_line[0:-1], laplace)
        if class_name == test_line[-1]:
            counter+=1
    assert_cross.append(counter/len(test))
print("Cross validation assert:", assert_cross)