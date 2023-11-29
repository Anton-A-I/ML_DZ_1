from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
import joblib

app = FastAPI()

loaded_model = joblib.load('trained_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
loaded_train_columns = joblib.load('trained_model_columns.pkl')


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

# item = {
#     "name": "Mazda",
#     "year": 2007,
#     "selling_price": 665000,
#     "km_driven": 25000,
#     "fuel": "Diesel",
#     "seller_type": "Individual",
#     "transmission": "Manual",
#     "owner": "First Owner",
#     "mileage": "21.5",
#     "engine": "1497 CC",
#     "max_power": "108.5 bhp",
#     "torque": "156 @ 5466846",
#     "seats": 5.00
#     }


# items = [
#     {
#     "name": "Mazda",
#     "year": 2007,
#     "selling_price": 665000,
#     "km_driven": 25000,
#     "fuel": "Diesel",
#     "seller_type": "Individual",
#     "transmission": "Manual",
#     "owner": "First Owner",
#     "mileage": "21.5",
#     "engine": "1497 CC",
#     "max_power": "108.5 bhp",
#     "torque": "156 @ 5466846",
#     "seats": 5.00
#     },
#     {
#     "name": "Honda",
#     "year": 2005,
#     "selling_price": 1000000,
#     "km_driven": 10000,
#     "fuel": "Diesel",
#     "seller_type": "Individual",
#     "transmission": "Manual",
#     "owner": "First Owner",
#     "mileage": "21.5",
#     "engine": "1900 CC",
#     "max_power": "200.0 bhp",
#     "torque": "156 @ 5466846",
#     "seats": 5.00
#     }
# ]
#
#
"""Предсказание цены одного объекта"""

def preprocess_input(item):
    df_test = pd.DataFrame(item, index=[0])
    df_test1 = df_test.copy()

    df_test1['engine'] = df_test1['engine'].replace(to_replace=r'[^\d?\.?\d]',
                                                    value='', regex=True)
    df_test1['engine'] = df_test1['engine'].astype(float)
    df_test1['mileage'] = df_test1['mileage'].replace(
        to_replace=r'[^\d?\.?\d]', value='', regex=True)
    df_test1['mileage'] = df_test1['mileage'].astype(float)
    df_test1['max_power'] = df_test1['max_power'].str.strip()
    df_test1['max_power'] = df_test1['max_power'].replace(
        to_replace=r'[^\d?\.?\d]', value='', regex=True)
    df_test1['max_power'] = df_test1['max_power'].replace('', '0')
    df_test1['max_power'] = df_test1['max_power'].astype(float)

    df_test1 = df_test1.drop('torque', axis=1)

    df_test1['engine'] = df_test1['engine'].astype(int)
    df_test1['seats'] = df_test1['seats'].astype(int)
    y_test = df_test1['selling_price'].copy()

    encoder = OneHotEncoder()

    X_test_cat = df_test1.drop(['selling_price', 'name'], axis=1)
    X_test_cat_encoded = encoder.fit_transform(
        X_test_cat[['fuel', 'seller_type', 'transmission', 'owner', 'seats']])
    X_test_cat_encoded_df = pd.DataFrame(X_test_cat_encoded.toarray(),
                                         columns=encoder.get_feature_names_out(
                                             ['fuel', 'seller_type',
                                              'transmission', 'owner',
                                              'seats']))
    X_test_cat_encoded = pd.concat(
        [X_test_cat[['year', 'km_driven', 'mileage', 'engine', 'max_power']],
         X_test_cat_encoded_df], axis=1)

    # scaler = MinMaxScaler()
    # X_train_cat_encoded[['year', 'km_driven', 'mileage', 'engine','max_power']] = scaler.fit_transform(X_train_cat_encoded[['year', 'km_driven', 'mileage', 'engine','max_power']])
    X_test_cat_encoded[['year', 'km_driven', 'mileage', 'engine',
                        'max_power']] = loaded_scaler.transform(
        X_test_cat_encoded[
            ['year', 'km_driven', 'mileage', 'engine', 'max_power']])
    missing_columns = set(loaded_train_columns) - set(
        X_test_cat_encoded.columns)
    for col in missing_columns:
        X_test_cat_encoded[col] = 0
    column_order = loaded_train_columns
    X_test_cat_encoded_1 = X_test_cat_encoded.reindex(columns=column_order,
                                                      fill_value=0)
    return X_test_cat_encoded_1


@app.post("/predict_item")
def predict_item(item: Item) ->float:
    test = preprocess_input(item)
    predicted_price = loaded_model.predict(test)
    return float(predicted_price)



"""Предсказание цен нескольких объектов"""
class Items(BaseModel):
    objects: List[Item]

# items = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_test.csv')

def preprocess_inputs(items):

    items = pd.DataFrame(items)
    df_test1 = items.copy()
    #
    df_test1['engine'] = df_test1['engine'].replace(to_replace=r'[^\d?\.?\d]',
                                                    value='', regex=True)
    df_test1['engine'] = df_test1['engine'].astype(float)
    df_test1['mileage'] = df_test1['mileage'].replace(
        to_replace=r'[^\d?\.?\d]', value='', regex=True)
    df_test1['mileage'] = df_test1['mileage'].astype(float)
    df_test1['max_power'] = df_test1['max_power'].str.strip()
    df_test1['max_power'] = df_test1['max_power'].replace(
        to_replace=r'[^\d?\.?\d]', value='', regex=True)
    df_test1['max_power'] = df_test1['max_power'].replace('', '0')
    df_test1['max_power'] = df_test1['max_power'].astype(float)

    df_test1 = df_test1.drop('torque', axis=1)
    median_mileage_test = df_test1['mileage'].median()
    median_engine_test = df_test1['engine'].median()
    median_max_power_test = df_test1['max_power'].median()
    median_seats_test = df_test1['seats'].median()
    df_test1['mileage'].fillna(median_mileage_test, inplace=True)
    df_test1['engine'].fillna(median_engine_test, inplace=True)
    df_test1['max_power'].fillna(median_max_power_test, inplace=True)
    df_test1['seats'].fillna(median_seats_test, inplace=True)

    df_test1['engine'] = df_test1['engine'].astype(int)
    df_test1['seats'] = df_test1['seats'].astype(int)
    y_test = df_test1['selling_price'].copy()

    encoder = OneHotEncoder()

    X_test_cat = df_test1.drop(['selling_price', 'name'], axis=1)
    X_test_cat_encoded = encoder.fit_transform(
        X_test_cat[['fuel', 'seller_type', 'transmission', 'owner', 'seats']])
    X_test_cat_encoded_df = pd.DataFrame(X_test_cat_encoded.toarray(),
                                         columns=encoder.get_feature_names_out(
                                             ['fuel', 'seller_type',
                                              'transmission', 'owner',
                                              'seats']))
    X_test_cat_encoded = pd.concat(
        [X_test_cat[['year', 'km_driven', 'mileage', 'engine', 'max_power']],
         X_test_cat_encoded_df], axis=1)


    X_test_cat_encoded[['year', 'km_driven', 'mileage', 'engine',
                        'max_power']] = loaded_scaler.transform(
        X_test_cat_encoded[
            ['year', 'km_driven', 'mileage', 'engine', 'max_power']])
    missing_columns = set(loaded_train_columns) - set(
        X_test_cat_encoded.columns)
    for col in missing_columns:
        X_test_cat_encoded[col] = 0
    column_order = loaded_train_columns
    X_test_cat_encoded_2 = X_test_cat_encoded.reindex(columns=column_order,
                                                      fill_value=0)
    return X_test_cat_encoded_2



@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    tests = preprocess_inputs(items)
    predicted_prices = loaded_model.predict(tests)
    return predicted_prices.tolist()

# print(predict_items(items))


