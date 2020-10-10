# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 18:36:33 2020

@author: quaxoc
"""

from datos import *
from test_train import *

dataset=Datos1('ConjuntosDatos/tic-tac-toe.data')
#print(dataset.nominalAtributos)
#print(dataset.diccionario)
rows_number=dataset.datos.shape[0]
test_proportion=0.3
line_ids=validacion_cruzada(rows_number,4)
line_ids_test=line_ids[0]['Test']
line_ids_train=line_ids[0]['Train']


print("Test",line_ids_test)
print("Train",line_ids_train)
print(dataset.extraeDatos(line_ids_test))