#Installing all necessary packages 
import smogn
import snowflake.connector
import numpy as np
import pandas as pd
import datetime as dt
import boto3
import base64
from numpy import dtype, isnan, sqrt
from sqlalchemy import create_engine
from sqlalchemy import pool
from sqlalchemy.dialects import registry
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives import serialization
from botocore.exceptions import ClientError
from snowflake.connector.pandas_tools import write_pandas, pd_writer
from dateutil.easter import *
from pathlib import Path
from imblearn.over_sampling import SMOTE
import gc
import dask.dataframe as dd
import dask
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier
import catboost
import time


import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from ltv_params import *

import warnings
warnings.filterwarnings("ignore")


def training_data(snowflake_table_train): 
    ctx = snowflake.connector.connect(
      user='TYLYNN',
      account='mja29153.us-east-1',
      password = 'Brenn1025!',
      warehouse='TABLEAU_L',
      database='CLIENT',
      schema='PROGRESSIVE'
      )
    cur = ctx.cursor()
    sql = """
    select
    LTV, 
    SUPPLY_VENDOR, 
    USER_HOUR_OF_WEEK, 
    REGION, 
    BROWSER, 
    DMA,
    PIQ_CHILD_PRESENT_AGE_GROUPS, 
    DWELLING, 
    ETHNIC_GROUP,
    PIQ_GENDER_GROUPS,
    PIQ_AGE_GROUPS_PRESENT,
    PIQ_AGE_GROUPS,
    PIQ_CHILDREN_ESTIMATE, 
    ETHNICIQ_V2,
    MARITAL,
    EDUCATION,
    PROPERTY_TYPE,
    PIQ_INCOME_GROUPS,
    PIQ_ADULT_ESTIMATE,
    PIQ_SPANISH_SPEAKING,
    PIQ_PEOPLE_IN_HH,
    PIQ_PRESENCE_OF_CHILDREN
    from "CLIENT"."PROGRESSIVE"."AUGUST_PIQ_JOIN_LTV_2"
    sample(10)
    
    
    """
    
    cur.execute(sql)
    df = cur.fetch_pandas_all()

    cur.close()
    ctx.close()
    
            
    return df

def fix_nulls_and_types(dataframe, nulls = 'Yes', types = 'Yes', val = 0, null_threshold=null_threshold):
    #dataframe = dataframe[pd.to_numeric(dataframe['ZIPCODE'], errors='coerce').notnull()]
    if nulls == "Yes":
        dataframe = dataframe.loc[:, dataframe.isnull().sum() < null_threshold * dataframe.shape[0]]
        dataframe = dataframe.fillna(value = val)
    if types == 'Yes':
        for col in dataframe.columns:
            if dataframe[col].dtype == 'float64' or dataframe[col].dtype == '<M8[ns]':
                dataframe[col] = dataframe[col].astype(str)
    return dataframe

def log_transform(dataframe):
    dataframe['LTV'] = dataframe['LTV'].astype(float)
    min_ltv = dataframe['LTV'].min()
    dataframe['LTV'] = dataframe['LTV']+1 #+ min_ltv

    dataframe['logltv'] = np.log(dataframe['LTV'])
    dataframe['logltv'].hist()
    dataframe = dataframe.dropna()
    X = dataframe.drop(['LTV', 'logltv'],1)
    Y = dataframe['logltv']
    return X, Y

def prediction_data(snowflake_table_pred): 
    ctx = snowflake.connector.connect(
      user='TYLYNN',
      account='mja29153.us-east-1',
      password = 'Brenn1025!',
      warehouse='TABLEAU_L',
      database='CLIENT',
      schema='PROGRESSIVE'
      )
    cur = ctx.cursor()
    sql = """

    select DEVICE_ADVERTISING_ID, SUPPLY_VENDOR,
     USER_HOUR_OF_WEEK,
     REGION,
     BROWSER,
     DMA,
     PIQ_CHILD_PRESENT_AGE_GROUPS,
     DWELLING,
     ETHNIC_GROUP,
     PIQ_GENDER_GROUPS,
     PIQ_AGE_GROUPS_PRESENT,
     PIQ_AGE_GROUPS,
     PIQ_CHILDREN_ESTIMATE,
     ETHNICIQ_V2,
     MARITAL,
     EDUCATION,
     PROPERTY_TYPE,
     PIQ_INCOME_GROUPS,
     PIQ_ADULT_ESTIMATE,
     PIQ_SPANISH_SPEAKING,
     PIQ_PEOPLE_IN_HH,
     PIQ_PRESENCE_OF_CHILDREN  from "CLIENT"."PROGRESSIVE"."PIQ_REDS_0811"
    OFFSET 0 ROWS FETCH NEXT 100000 ROWS ONLY

      """
    cur.execute(sql)
    df_piq = cur.fetch_pandas_all()

    cur.close()
    ctx.close()
    
    return df_piq

def catboost_prediction(df_piq, clf, X):
    cols = X.columns.tolist()
    df_piq.fillna('0', inplace=True)
    df_piq['USER_HOUR_OF_WEEK'] = df_piq['USER_HOUR_OF_WEEK'].astype(float).astype(int).round(0)
    df_piq['BROWSER'] = df_piq['BROWSER'].astype(float).astype(int).round(0)
    Z = df_piq.drop(['DEVICE_ADVERTISING_ID'],1)
    Z = Z[cols]
    print(Z)
    vals = clf.predict(Z)
    df_ltv = pd.DataFrame({'LTV': vals})
    df_piq['ltv'] = df_ltv['LTV']
    df_piq.to_csv('chalice_test_daids.csv')
    return df_piq



    



    
