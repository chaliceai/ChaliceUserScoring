#Installing all necessary packages 
import snowflake.connector
import numpy as np
import pandas as pd
import datetime as dt
# from dateutil.easter import *
from ChaliceUserScoring.Utils.TTD_user_score_upload import *

import warnings
warnings.filterwarnings("ignore")

"""
should take in :
    traininng table name
    training table attributes list
"""
def training_data(cur, snowflake_table_train, table_attributes,sample_size=None): 

    attributes = ', '.join(table_attributes)
    sql = f"SELECT {attributes} FROM {snowflake_table_train} LIMIT 1000" #LIMIT ADDED FOR TESTING
    if(sample_size != None):
        sql = sql + f"sample({sample_size})"

    cur.execute(sql)
    df = cur.fetch_pandas_all()
    return df

#looks good
def fix_nulls_and_types(dataframe, null_threshold, nulls = 'Yes', types = 'Yes', val = 0):
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
    