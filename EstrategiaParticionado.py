from abc import ABCMeta,abstractmethod


class Particion():

  # Esta clase mantiene la lista de índices de Train y Test para cada partición del conjunto de particiones  
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor 
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta  
  def creaParticiones(self,datos,seed=None):
    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

  def __init__(self, test_proportion, n_iters)
    self.test_proportion=test_proportion
    self.n_iters = n_iters
  
  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el número de ejecuciones deseado
  # Devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):
    random.seed(seed)
    id_list=list(range(len(datos))) #datos es una lista de las filas de la tabla, cada fila siendo otra lista
    ret = []
    for i in range(self.n_iters):
      random.shuffle(id_list)
      cut=round(self.test_proportion*len(datos))
      aux = Particion()
      aux.indicesTest = sorted(id_list[0:cut])
      aux.indicesTrain = sorted(id_list[cut:])
      ret.append(aux)
    return ret
      
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
  
  def __init__(self, ngrupos)
    self.ngrupos = ngrupos #numero de divisiones de los datos
  
  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):   
    random.seed(seed)
    id_list=list(range(len(datos))) #datos es una lista de las filas de la tabla, cada fila siendo otra lista
    n, m = divmod(len(datos), self.ngrupos)
    
    random.shuffle(id_list)
    #print(id_list)
    ret=[]
    chunks=[]
    for i in range(self.ngrupos):
      chunks.append(sorted(id_list[i*n+min(i,m):(i+1)*n+min(i+1,m)]))
    #print(chunks)

    for i in range(self.ngrupos):
      train=chunks.copy()
      test=train.pop(i)
      train1d=[]
      for ch in train:
        train1d+=ch
      aux = Particion()
      aux.indicesTest = test
      aux.indicesTrain = sorted(train1d)
      ret.append(aux)
      #sampling.append(validacion_simple(row_count,test_proportion))
    return ret
