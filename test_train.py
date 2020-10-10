# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 19:27:06 2020

@author: quaxoc
"""
import random 
import numpy as np

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
        

row_count=21
test_proportion=0.3

'''
print("Validacion simple")
print(validacion_simple(row_count,test_proportion))
k=4
'''


#print(validacion_cruzada(row_count,4))
