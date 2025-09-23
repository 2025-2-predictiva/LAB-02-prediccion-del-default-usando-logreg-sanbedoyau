# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pickle
import json
import os
import gzip
from glob import glob
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, compression = 'zip')
    return df

def clean_data(DataFrame: pd.DataFrame) -> pd.DataFrame:
    DataFrame.drop(columns = 'ID', inplace = True)
    DataFrame.rename(columns = {'default payment next month': 'default'},
                     inplace = True)
    DataFrame['EDUCATION'] = DataFrame['EDUCATION'].apply(lambda x: 4 if x >= 4 else x).astype('category')
    DataFrame = DataFrame.query('EDUCATION != 0 and MARRIAGE != 0')
    return DataFrame

def features_target_split(DataFrame: pd.DataFrame) -> tuple:
    return DataFrame.drop(columns = 'default'), DataFrame['default']

def make_pipeline(estimator: LogisticRegression, cat_features: list) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers = [
            ('ohe', OneHotEncoder(dtype = 'int'), cat_features)
        ],
        remainder = MinMaxScaler()
    )

    selectKBest = SelectKBest(f_regression)

    pipeline = Pipeline(
        steps = [
            ('preprocessor', preprocessor),
            ('selectkbest', selectKBest),
            ('regressor', estimator)
        ],
        verbose = False
    )

    return pipeline

def make_grid_search(estimator: Pipeline, param_grid: dict, cv = 10) -> GridSearchCV: 
    grid_search = GridSearchCV(
        estimator = estimator,
        param_grid = param_grid,
        cv = cv,
        scoring = 'balanced_accuracy'
    )
    return grid_search

def save_estimator(path: str, estimator: GridSearchCV) -> None:
    with gzip.open(path, 'wb') as file:
        pickle.dump(estimator, file)

def eval_model(estimator: GridSearchCV, features: pd.DataFrame, target: pd.Series, name: str) -> dict:
    y_pred = estimator.predict(features)
    metrics = {
        'type': 'metrics',
        'dataset': name,
        'precision': precision_score(target, y_pred),
        'balanced_accuracy': balanced_accuracy_score(target, y_pred),
        'recall': recall_score(target, y_pred),
        'f1_score': f1_score(target, y_pred)
    }
    return metrics
    
def save_metrics(path: str, train_metrics: dict, test_metrics: dict) -> None:
    with open(path, 'w') as file:
        file.write(json.dumps(train_metrics) + '\n')
        file.write(json.dumps(test_metrics) + '\n')

def confusion_mtrx(estimator: GridSearchCV, features: pd.DataFrame, target: pd.Series, name: str) -> dict:
    y_pred = estimator.predict(features)
    cm = confusion_matrix(target, y_pred)
    mtrx = {
        'type': 'cm_matrix',
        'dataset': name,
        'true_0': {'predicted_0': int(cm[0, 0]),
                   'predicted_1': int(cm[0, 1])},
        'true_1': {'predicted_0': int(cm[1, 0]),
                   'predicted_1': int(cm[1, 1])}
    }
    return mtrx

def save_cm(path: str, train_mtrx: dict, test_mtrx: dict) -> None:
    with open(path, 'a') as file:
        file.write(json.dumps(train_mtrx) + '\n')
        file.write(json.dumps(test_mtrx))

def create_out_dir(out_dir: str) -> None:
        if os.path.exists(out_dir):
            for file in glob(f'{out_dir}/*'):
                os.remove(file)
            os.rmdir(out_dir)
        os.makedirs(out_dir)


def run():
    in_path = 'files/input'
    out_path = 'files/output'
    mod_path = 'files/models'
    train = clean_data(load_data(f'{in_path}/train_data.csv.zip'))
    test = clean_data(load_data(f'{in_path}/test_data.csv.zip'))
    
    x_train, y_train = features_target_split(train)
    x_test, y_test = features_target_split(test)
    
    cat_features = [cat for cat in x_train if x_train[cat].dtype == 'category']
    estimator = make_pipeline(LogisticRegression(), cat_features)

    param_grid = {
        'selectkbest__k': range(1, len(x_train.columns) + 1)
    }
    estimator = make_grid_search(estimator, param_grid = param_grid)
    
    estimator.fit(x_train, y_train)

    create_out_dir(f'{out_path}')
    create_out_dir(f'{mod_path}')

    save_estimator(f'{mod_path}/model.pkl.gz', estimator)

    train_metrics = eval_model(estimator, x_train, y_train, 'train')
    test_metrics = eval_model(estimator, x_test, y_test, 'test')
    save_metrics(f'{out_path}/metrics.json', train_metrics, test_metrics)

    train_cm = confusion_mtrx(estimator, x_train, y_train, 'train')
    test_cm = confusion_mtrx(estimator, x_test, y_test, 'test')
    save_cm(f'{out_path}/metrics.json', train_cm, test_cm)


if __name__ == '__main__':
    run()