#Installing all necessary packages 
import snowflake.connector
import numpy as np
import pandas as pd
import datetime as dt
# from dateutil.easter import *
from ltv_params import *
from TTD_user_score_upload import *

import warnings
warnings.filterwarnings("ignore")

"""
should take in :
    traininng table name
    training table attributes list
"""
def training_data(snowflake_table_train, table_attributes,sample_size=None): 
    ctx = snowflake.connector.connect(
      user='TYLYNN',
      account='mja29153.us-east-1',
      password = 'Brenn1025!',
      warehouse='TABLEAU_L',
      database='CLIENT',
      schema='PROGRESSIVE'
      )
    cur = ctx.cursor()
    attributes = table_attributes.join(',')
    
    sql = f"SELECT {attributes} FROM {snowflake_table_train}"
    if(sample_size != None):
        sql = sql + f"sample({sample_size})"

    cur.execute(sql)
    df = cur.fetch_pandas_all()
    cur.close()
    ctx.close()     
    return df

#looks good
def fix_nulls_and_types(dataframe, nulls = 'Yes', types = 'Yes', val = 0, null_threshold=null_threshold):
    if nulls == "Yes":
        dataframe = dataframe.loc[:, dataframe.isnull().sum() < null_threshold * dataframe.shape[0]]
        dataframe = dataframe.fillna(value = val)
    if types == 'Yes':
        for col in dataframe.columns:
            if dataframe[col].dtype == 'float64' or dataframe[col].dtype == '<M8[ns]':
                dataframe[col] = dataframe[col].astype(str)
    return dataframe


#looks good
def log_transform(dataframe):
    dataframe['LTV'] = dataframe['LTV'].astype(float)
    min_ltv = dataframe['LTV'].min()
    dataframe['LTV'] = dataframe['LTV'] + min_ltv

    dataframe['logltv'] = np.log(dataframe['LTV'])
    dataframe['logltv'].hist()
    dataframe = dataframe.dropna()
    X = dataframe.drop(['LTV', 'logltv'],1)
    Y = dataframe['logltv']
    return X, Y

"""
NEEDS:
    prediction table name
    predictino table attribute list
"""
def prediction_data(snowflake_table_pred, table_attributes, offset=0): 
    ctx = snowflake.connector.connect(
      user='TYLYNN',
      account='mja29153.us-east-1',
      password = 'Brenn1025!',
      warehouse='TABLEAU_L',
      database='CLIENT',
      schema='PROGRESSIVE'
      )
    cur = ctx.cursor()
    attrbutes = table_attributes.join(',')
    # offset_statement = f'OFFSET {offset} ROWS FETCH NEXT 100000 ROWS ONLY'
    # sql = f"SELECT {attrbutes} FROM {snowflake_table_pred} {offset_statement}"
    sql = f"SELECT {attrbutes} FROM {snowflake_table_pred}"
    # look into fetch pandas batches
    cur.execute(sql)
    df_piq = cur.fetch_pandas_all()

    cur.close()
    ctx.close()
    
    return df_piq

#looks good
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
    #df_piq.to_csv('chalice_test_daids.csv')
    return df_piq
    