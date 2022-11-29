# from ChaliceAPIUsage.APIConnection import *
from ChaliceUserScoring.Utils.Catboost_SMOTE import testtrainsplit, catboost_regression
from ChaliceUserScoring.Utils.ltv_model import *
from ChaliceUserScoring.Utils.TTD_user_score_upload import *
import warnings
import time
warnings.filterwarnings("ignore")


"""
NEEDS: 
    Table Name   type: string
    Attributes to select    type: list of strings
    null_treshold   type: float

DOES:
    Gets data from table as a df
    formats data to be usable
    returns df w/ data, and 2 more df's with specific data
"""
def data_prep(cur, snowflake_table_train_name, train_table_attributes, 
              null_threshold):

    df = training_data(cur, snowflake_table_train_name, train_table_attributes)

    dataframe = fix_nulls_and_types(df, null_threshold, nulls='Yes', types='Yes',
                                    val=0)
    X, Y= log_transform(dataframe)
    
    return df, X, Y

"""
NEEDS:
    X type: df
    Y type: df
    test_proportion type: float

DOES:
    Trains and runs catboost on given df's
    returns output of catboost regression
 
*Needs import Catboost_SMOTE. testtrainsplit, catboost_regression
"""       
def train_model(X, Y, test_proportion, bootstrap_type, 
                depth, learning_rate, loss_function, iteration):

    X_train, X_val, Y_train, Y_val = testtrainsplit(X,Y,test_proportion=test_proportion)

    clf, categorical_features_indices = catboost_regression(X_train, X_val, Y_train,
                                                            Y_val, bootstrap_type=bootstrap_type,
                                                            depth=depth,learning_rate=learning_rate,
                                                            loss_function=loss_function, iteration=iteration)
    return clf, categorical_features_indices



    
    
def model_prediction(snowflake_table_pred_name, table_attributes, clf, X,
                     bucket, prefix, csv_name, cur):
    attrbutes = ', '.join(table_attributes)
   
    sql = f"SELECT {attrbutes} FROM {snowflake_table_pred_name} LIMIT 1000" #ADDED LIMIT FOR TESTING
    cur.execute(sql)
    list_of_csvs = []
    for i, df in enumerate(cur.fetch_pandas_batches()):
        df = catboost_prediction(df, clf, X)
        df = daid_format_pandas(df)
        csv_name = csv_name.removesuffix('.csv')
        df = pandas_push_to_s3(df, bucket, prefix, f"{csv_name}_{i + 1}_{dt.datetime.utcnow().strftime('%Y-%m-%d')}.csv")
        list_of_csvs.append(df)

    return list_of_csvs