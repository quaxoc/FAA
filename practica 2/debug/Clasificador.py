from abc import ABCMeta,abstractmethod
from EstrategiaParticionado import *
import numpy as np
from scipy.stats import norm


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
  def error(self,datos,pred): #datos es el retorno de extraeDatos de los testIds, pred es una lista de retornos del metodo clasifica
    counter=0
    i=0
    for test_line in datos:
      if pred[i] != test_line[-1]:
        counter+=1
      i+=1
    error=counter/len(datos) #aqui ponia len(test), no se de donde salia la variable test
    return(error)
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset,clasificador,seed=None):
       
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test. Otra opcion es repetir la validacion simple un numero especificado de veces, obteniendo en cada una un error. Finalmente se calcularia la media.
    if isinstance(particionado, ValidacionSimple):
      line_ids=particionado.creaParticiones(dataset.datos)
      assert_cross=[]
      for i in range(particionado.n_iters):
        line_ids_test=line_ids[i].indicesTest
        line_ids_train=line_ids[i].indicesTrain
        train=dataset.extraeDatos(line_ids_train)
        test=dataset.extraeDatos(line_ids_test)
        predictions=[]
        for test_line in test:
          clasificador.entrenamiento(train, dataset.nominalAtributos, dataset.diccionario)
          predictions.append(clasificador.clasifica(test_line[0:-1], dataset.nominalAtributos, dataset.diccionario))
        assert_cross.append(clasificador.error(test, predictions))
      ret=[]
      ret.append(np.mean(assert_cross))
      ret.append(np.std(assert_cross))
      return ret
      
    elif isinstance(particionado, ValidacionCruzada):
      line_ids=particionado.creaParticiones(dataset.datos)
      assert_cross=[]
      for i in range(particionado.ngrupos):
        line_ids_test=line_ids[i].indicesTest
        line_ids_train=line_ids[i].indicesTrain
        train=dataset.extraeDatos(line_ids_train)
        test=dataset.extraeDatos(line_ids_test)
        predictions=[]
        for test_line in test:
          clasificador.entrenamiento(train, dataset.nominalAtributos, dataset.diccionario)
          predictions.append(clasificador.clasifica(test_line[0:-1], dataset.nominalAtributos, dataset.diccionario))
        assert_cross.append(clasificador.error(test, predictions))
      ret=[]
      ret.append(np.mean(assert_cross))
      ret.append(np.std(assert_cross))
      return ret


##############################################################################
class ClasificadorVecinosProximos(Clasificador):
  def __init__(self,classes,k,dist_type):
    #self.train=train_set.copy(deep=True)
    self.classes=classes.values
    self.class_names=np.unique(self.classes)
    self.k=k
    self.dist_type=dist_type
    self.dists=[]
    self.normalize_done=False
    self.calcularMediasDesv()
   
  def calcularMediasDesv(self,datos,nominalAtributos): #datos son los datos de entrenamiento
    self.mean=datos.mean(axis=0)
    self.std=datos.std(axis=0)
  
  def normalizarDato(self, dato):
    x=(dato-self.mean.values)/self.std.values
    return(x)
  def normalizarDatos(self,datos,nominalAtributos):
    if not self.normalize_done:
      for att in datos.columns: #datos son los datos de entrenamiento
        datos[att]=(datos[att]-self.mean[att])/self.std[att]
        self.normalize_done=True
    return(self.train)

  def euclidian_distance(self,dato):
    i=0
    dist=0
    for att in self.train.columns:
      dist=dist+(self.train[att].values-dato[i])**2
      i+=1
    dist=dist**0.5
    self.dists.append(dist)
    return(dist)
  def manhatten_distance(self,dato):
    i=0
    dist=0
    for att in self.train.columns:
      dist=dist+(self.train[att].values-dato[i])
      i+=1
    dist=abs(dist)
    self.dists.append(dist)
    return(dist)
  def mahalanobis_distance(self,dato): 
    X=self.train.values
    if self.normalize_done:
      mu=np.zeros(len(dato))
    else:
      mu=self.mean
    cov=np.cov(np.transpose(X-mu))
    diff=dato - mu
    inv=np.linalg.inv(cov)
    left=np.dot(diff,inv)
    dist=np.dot(left,np.transpose(diff)) #supposed to be a scalar
    self.dists.append(dist)
    return(dist)
  
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario): #datostrain es el retorno de extraeDatos de los ids de entrenamiento
    self.train=datostrain.copy(deep=True)
  
  def kneignour_class(self):
    idx = np.argpartition(self.dist, self.k)
    idx=idx[0:self.k]
    return mode(self.classes[idx])
  def clasifica(self,datostest,atributosDiscretos,diccionario): #datostest es un elemento del retorno de extraeDatos de los ids de test
    predicted_class=[]
    if self.dist_type=="manhatten":
      for x in datostest:
        dist=self.manhatten_distance(x)
        idx = np.argpartition(dist, self.k)
        idx=idx[0:self.k]
        predicted_class.append(mode(self.classes[idx]))
    elif self.dist_type=="mahalanobis":
      for x in datostest:
        dist=self.mahalanobis_distance(x)
        idx = np.argpartition(dist, self.k)
        idx=idx[0:self.k]
        predicted_class.append(mode(self.classes[idx]))
    elif self.dist_type=="euclidian":
      for x in datostest:
        dist=self.euclidian_distance(x)
        idx = np.argpartition(dist, self.k)
        idx=idx[0:self.k]
        predicted_class.append(mode(self.classes[idx]))
    return(predicted_class)
  

class ClasificadorNaiveBayes(Clasificador):

  def __init__(self, laplace):
    self.prob_dada_clase=[]
    self.p_priori=[]
    self.laplace=laplace

  # TODO: implementar
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario): #datostrain es el retorno de extraeDatos de los ids de entrenamiento
    #The list of classes with ids
    classes=diccionario[-1]
    #The list of atributes with ids
    atr=diccionario[:-1]
    
    train_classes=datostrain[:, -1]
    for c in classes:
        self.p_priori.append(np.sum(train_classes==c)/len(train_classes))
    
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
        if (0 in p_table and self.laplace):
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
    
    
    
  # TODO: implementar
  def clasifica(self,datostest,atributosDiscretos,diccionario): #datostest es un elemento del retorno de extraeDatos de los ids de test
    #The list of classes with ids
    classes=diccionario[-1]
    #The list of atributes with ids
    atr=diccionario[:-1]
    
    #Convert test line to ids using the dictionary
    test_line_ids=[]
    for k in range(len(datostest)):
      if atributosDiscretos[k]:
        test_line_ids.append((atr[k][datostest[k]]))
      else:
        test_line_ids.append(datostest[k])
    
    P_posteriori=[]
    for c_i in range(len(classes)):
      j=0
      p_c_post=self.p_priori[c_i]
      for atr_val in test_line_ids:
        if atributosDiscretos[j]:
          p_c_post=p_c_post*self.prob_dada_clase[j][atr_val,c_i]
        else:
          mu=self.prob_dada_clase[j][0,c_i]
          std=self.prob_dada_clase[j][1,c_i]
          p_contin=norm.pdf(atr_val, mu, std)
          p_c_post=p_c_post*p_contin
        j+=1
      P_posteriori.append(p_c_post)
     
    P_posteriori=P_posteriori/np.sum(P_posteriori)    
    index_max = np.argmax(P_posteriori)
    
    #Return predicted class 
    predicted_class=list(classes.keys())[list(classes.values()).index(index_max)]
    #return(predicted_class, P_posteriori)
    return(predicted_class)