# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:58:41 2020

@author: quaxoc
"""

import numpy as np
from sklearn import preprocessing
from datos import *

dataset=Datos('ConjuntosDatos/tic-tac-toe.data')
data_table=dataset.datos
le = preprocessing.LabelEncoder()
a=np.empty([data_table.shape[0], data_table.shape[1]])
for i in range(data_table.shape[1]):
    column=data_table[:,i]
    column=le.fit_transform(column)
    a[:,i]=np.transpose(column)

print(a.shape)   
print(a) 