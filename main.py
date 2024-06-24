from fastapi import FastAPI, Query
import json

import numpy as np
import pandas as pd
import datetime as dt
from random import sample
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

app.title = "Aplicación CREDIKG"

'''Función para calcular la edad'''
def calcular_edad(fecha_nac):
    hoy = dt.datetime.now()
    edad = hoy.year - fecha_nac.year - ((hoy.month, hoy.day) < (fecha_nac.month, fecha_nac.day))
    return edad

'''Random Forest'''
@app.get('/mr', tags=['Home'])
def get_mr(data: str = Query(...)):
    #Convertir el string JSON a un arreglo
    solicitud = json.loads(data)

    #Lectura del csv
    datos_credito = pd.read_csv('Layout.csv')

    #Elimina filas con valores NaN
    datos_credito = datos_credito.dropna()

    #Convertir la columna de Birth Date a tipo datetime
    datos_credito['Birth Date'] = pd.to_datetime(datos_credito['Birth Date'])
    
    #Aplicar la función a la columna de fecha de nacimiento
    datos_credito['Birth Date'] = datos_credito['Birth Date'].apply(calcular_edad)
    
    #Creación del bosque
    forest = RandomForestClassifier(n_estimators=100,
                                criterion="gini",
                                max_features="sqrt",
                                bootstrap=True,
                                max_samples=2/3,
                                oob_score=True)

    #Alimentación del bosque
    forest.fit(datos_credito.drop("Payment Status", axis='columns').values,datos_credito["Payment Status"])
    
    #Prueba del bosque
    #solicitud=[5,3360,2,2.0,5,'1966-02-01',1,4,654,45000,60000,10000]
    solicitud[5] = calcular_edad(dt.datetime.strptime(solicitud[5], '%Y-%m-%d'))

    respuesta = forest.predict([solicitud])
    
    return int(respuesta[0])

'''Modelo de Regresion Lineal'''
@app.get('/cc/{score}', tags=['Home'])
def get_cc(score: int):
    #Lectura del csv
    datos_credito = pd.read_csv('insumo_cc.csv')

    #Selección de Datos
    datos_credito = datos_credito.query('Score > 550')

    #Entrenamiento
    #Se partira el dataframe en 2 por datos de entrenamiento y datos de test
    datos_entrenamiento= datos_credito.sample(frac=0.8,random_state=0)
    datos_test=datos_credito.drop(datos_entrenamiento.index)

    #Separar la variable que queremos predecir(En este caso el Monto a prestar)
    etiquetas_entrenamiento=datos_entrenamiento.pop('Monto')
    etiquetas_test = datos_test.pop('Monto')

    #Entrenamiento de una regresion lineal
    modelo = LinearRegression()
    modelo.fit(datos_entrenamiento,etiquetas_entrenamiento)

    #Nuevo solicitante
    nuevo_solicitante = pd.DataFrame(np.array([[score]]), columns=['Score'])

    respuesta = modelo.predict(nuevo_solicitante)
    return int(respuesta[0])