from abc import ABCMeta,abstractmethod

from datos import *
import pandas as pd
import numpy as np
import random 
from scipy.stats import norm
from EstrategiaParticionado import *


class Clasificador:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
    pass
  
  
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto
  # devuelve un numpy array con las predicciones
  def clasifica(self,datosTest,atributosDiscretos,diccionario):
    pass
  
  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # TODO: implementar
  def error(self,datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error    
    i=0
    ret=0.0
    for elem in pred:
      if elem == datos[i]:
        ret+=1
      i+=1
    return ret/i
	#pass
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset,clasificador,seed=None):
       
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test. Otra opcion es repetir la validacion simple un numero especificado de veces, obteniendo en cada una un error. Finalmente se calcularia la media.
    
    particiones=particionado.creaParticiones(dataset) #datos es una lista de las filas de la tabla, cada fila siendo otra lista
    if(isinstance(particionado, ValidacionSimple)):
      line_ids=particionado.creaParticiones(dataset)
      line_ids_test=[] #line_ids[o.indicesTest for o in line_ids]
      line_ids_train=[] #line_ids[o.indicesTrain for o in line_ids]
      for o in line_ids:
        line_ids_test.append(o.indicesTest)
        line_ids_train.append(o.indicesTrain)

      train=dataset.extraeDatos(line_ids_train)
      test=dataset.extraeDatos(line_ids_test)

      counter=0
      for test_line in test:
        clasificador.entrenamiento(train, test_line[0:-1], dataset.diccionario) #no se si esto es correcto, atributosDiscretos?
        class_name, P_classes = clasificador.clasifica(test, test_line[0:-1], dataset.diccionario)
        if class_name == test_line[-1]:
          counter+=1
      assert_simple=counter/len(test)
      return error(test, assert_simple)
    elif(isinstance(particionado, ValidacionCruzada)):
      line_ids=particionado.creaParticiones(dataset)
      assert_cross=[]
      for i in range(particionado.ngrupos): 
        line_ids_test=line_ids[i].indicesTest
        line_ids_train=line_ids[i].indicesTrain
        train=dataset.extraeDatos(line_ids_train)
        test=dataset.extraeDatos(line_ids_test)
        counter=0
        for test_line in test:
          clasificador.entrenamiento(train, test_line[0:-1], dataset.diccionario)
          class_name, P_classes = clasificador.clasifica(test, test_line[0:-1], dataset.diccionario)
          if class_name == test_line[-1]:
            counter+=1
        assert_cross.append(counter/len(test))
      return error(test, assert_cross)
    else:
      print("Unknown validation method")

##############################################################################

class ClasificadorNaiveBayes(Clasificador):
  prob_dada_clase = []
  P_posteriori = []
  predicted_class = []
  p_priori = []

  # TODO: implementar
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
    #The list of classes with ids
    classes=diccionario[-1]
    #The list of atributes with ids
    atr=diccionario[:-1]
    
    train_classes=datostrain[:, -1]
    for c in classes:
        self.p_priori.append(np.sum(train_classes==c)/len(train_classes))
    
    
    #An array with tables for each attribute that either contains probabilities or mean and std for non categorical values
    a_id=0
    for a in atr:
      if atributosDiscretos[a_id]:
        p_table=np.empty([len(a), len(classes)])  #Creating a table that has probabilities for each atribute (rows) considering the class (columns)
        i=0
        for c in classes:
          j=0
          for aval in a.keys():
            p_table[j,i]=datostrain[np.where((datostrain[:,-1]==c) & (datostrain[:,a_id]==aval))].shape[0]
            j+=1
          i+=1
          #aplicando Laplace
          if (0 in p_table): # "and laplace" eliminado de la condicion
            p_table=p_table+1
          #Convirtiendo occuriencias en probabilidades
          p_table/=np.sum(p_table, axis=0)
      else:   #Apply gaussian bayes
        p_table=np.empty([2, len(classes)])  #Each column of the table is a class, first row is mean and second is standard deviation of the data values
        i=0
        for c in classes:
          p_table[0,i]=np.mean(datostrain[np.where(datostrain[:,-1]==c)][:, a_id])
          p_table[1,i]=np.std(datostrain[np.where(datostrain[:,-1]==c)][:, a_id])
          i+=1
      self.prob_dada_clase.append(p_table)
      a_id+=1
    return self.prob_dada_clase
  
  
  # TODO: implementar
  def clasifica(self,datostest,atributosDiscretos,diccionario):
    #The list of classes with ids
    classes=diccionario[-1]
    #The list of atributes with ids
    atr=diccionario[:-1]
    
    auxprob = np.asarray(self.prob_dada_clase)
    for c_i in range(len(classes)):
      j=0
      p_c_post=self.p_priori[c_i]
      for atr_val in datostest:
        aux_atr = int(atr_val)
        if atributosDiscretos[j]:
          p_c_post=p_c_post*auxprob[j][aux_atr,c_i] #self.prob_dada_clase[j][atr_val,c_i]
        else:
          mu=auxprob[j][0,c_i] #self.prob_dada_clase[j][0,c_i]
          std=auxprob[j][1,c_i] #self.prob_dada_clase[j][1,c_i]
          p_contin=norm.pdf(aux_atr, mu, std)
          p_c_post=p_c_post*p_contin
        j+=1
      self.P_posteriori.append(p_c_post)
     
    self.P_posteriori=self.P_posteriori/np.sum(self.P_posteriori)    
    index_max = np.argmax(self.P_posteriori)
    
    #Return predicted class 
    self.predicted_class=list(classes.keys())[list(classes.values()).index(index_max)]
    return(self.predicted_class, self.P_posteriori)