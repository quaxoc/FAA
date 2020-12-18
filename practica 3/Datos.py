# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 19:11:01 2020

@author: Evgeniia Makarova
"""

import pandas as pd
import numpy as np

class Datos:
  
  # TODO: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionario
  def __init__(self, filename):

    data_tic = pd.read_csv(filename)
    #Array of column names
    self.column_names=list(data_tic.columns)
    #Converting data from panda dataframe to numpy array
    self.datos=data_tic.to_numpy()

    #Checking column data types where strings are returned as True, interger and float as false 
    #and an exception for other data types 
    self.nominalAtributos=[]
    for val in self.datos[0,:]:
        if isinstance(val, str):
            self.nominalAtributos.append(True)
        elif isinstance(val, int) or isinstance(val, float):
            self.nominalAtributos.append(False)
        else:
            raise TypeError("Data types can only include string, interger or float")
    
    #getting the number of columns
    columns = self.datos.shape[1]
    #creating a dictionary as an array of dictionaries for each column
    self.diccionario=[]
    for i in range(columns):
        dict_tmp={}
        #filling dictionary with unique values
        values=np.unique(self.datos[:,i])
        dict_tmp=dict(zip(values,range(len(values))))
        self.diccionario.append(dict_tmp)

    
  # TODO: implementar en la práctica 1
  def extraeDatos(self, idx):
      test_data=self.datos[idx]
      return(test_data)


class Datos1:
  
  # TODO: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionario
    def __init__(self, filename):
        #filename='ConjuntosDatos/tic-tac-toe.data'
        data_tic = pd.read_csv(filename)
        column_names=list(data_tic.columns)
        self.datos=data_tic.to_numpy()
        
        self.nominalAtributos=[]
        for val in self.datos[0,:]:
            if isinstance(val, str):
                self.nominalAtributos.append(True)
            elif isinstance(val, int) or isinstance(val, float):
                self.nominalAtributos.append(False)
            else:
                raise TypeError("Data types can only include string, interger or float")
        
        columns = self.datos.shape[1]
        self.diccionario={}
        for i in range(columns):
            dict_tmp={}
            values=np.unique(self.datos[:,i])
            dict_tmp=dict(zip(values,range(len(values))))
            self.diccionario[column_names[i]]=dict_tmp

    def extraeDatos(self, idx):
        test_data=self.datos[idx]
        return(test_data)
   


"""
dataset=Datos('ConjuntosDatos/tic-tac-toe.data')
print(dataset.nominalAtributos)
print(dataset.diccionario)
print(dataset.datos)

dataset2=Datos1('ConjuntosDatos/tic-tac-toe.data')
print(dataset2.diccionario)
"""