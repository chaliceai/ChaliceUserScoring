# from ChaliceAPIUsage.APIConnection import *
from Utils.Catboost_SMOTE import *
from Utils.ltv_model import *
from Utils.TTD_user_score_upload import *
from ltv_params import *
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
def data_prep(snowflake_table_train_name, train_table_attributes, 
              null_threshold):

    df = training_data(snowflake_table_train_name, train_table_attributes)

    dataframe = fix_nulls_and_types(df, nulls='Yes', types='Yes',
                                    val=0, null_threshold=null_threshold)
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

"""
NEEDS:
    Table Name   type: string
    Attributes to select    type: list of strings
    clf from train_model type: ??
    X from data_prep

DOES:
    runs model prediction
"""
def model_prediction(snowflake_table_pred_name, pred_table_attributes, clf, X):
    df_piq = prediction_data(snowflake_table_pred_name, pred_table_attributes)
    df_piq = catboost_prediction(df_piq, clf, X)
    return df_piq


"""
NEEDS:
    csv file name   type: string
    bucket name type: string
    prefix  type: string

Does:
    formats csv
    pushes csv to s3
"""
def format_results(csv_name):
    daids = daid_format(csv_name = csv_name)
    push_to_s3(csv_name=csv_name, bucket=bucket, prefix=prefix)
    return daids


"""
NEEDS:
    Number of files?
    User for chaliceapiusage
    advertiser_id for post req
    csv file
    *** segment_name='LTVUploadOctober') add timetamp
DOES:
    posts csv file to ttd

pandas chunks
"""
def push_to_TTD(USER, ADVERTISER_ID , file_name, segment_name, NUM_FILES=1,):
    from ChaliceAPIUsage.APIConnection import TradedeskAPIConnection
    conn = TradedeskAPIConnection(USER)
    conn.set_secret(advertiser_id=ADVERTISER_ID,
                    secret_key='uar6lmos5mfccyjb26yhcepab3bmccel') #needs to be generalized

    # s3://tradedesk-uploads/daids_user_scores/chalice_demo/chalice_test_daids.csv'
    failed_files = 0
    t1 = time.time()
    results = []
    for i in range(NUM_FILES):
        print(f'Uploading chunk {i + 1}/{NUM_FILES}')
        print(file_name.format(i + 1))
        try:
            result = conn.post_data(advertiser_id=ADVERTISER_ID,
                                scores_csv=file_name.format(i + 1),
                                segment_name=segment_name)
        except UnicodeDecodeError as e:
            print(f'**FAILED** Unable to read {file_name.format(i + 1)}. Skipping...')
            failed_files += 1
            continue
        except Exception as e:
            print(f'**FAILED** Upload of {file_name.format(i + 1)} failed due to {type(e)}. Skipping...')
            failed_files += 1
            raise e
        results.append(result)
    t2 = time.time()
    print(f'Success! Completed at: {dt.datetime.now()}. Time elapsed: {t2 - t1}')
    print('Total IDs Submitted:', sum([x["TotalIDs"] for x in results]))
    print('Total Lines with Errors:', sum([x['FailedIDs'] for x in results]))
    print(f'Number of files skipped due to error: {failed_files}')
    

    
"""
Ideal Function:

    takes in
        training table name
        training table attributes list
        null threshold float
        test proportion float
        prediction table name
        predictioin table attributes list

    runs 
        data_prep
        train_model
        model_prediction
        format_results
        push to s3

    does not:
        push to ttd
"""

def userScoring(snowflake_table_train_name, train_table_attributes, 
                null_threshold, test_proportion, bootstrap_type, 
                depth, learning_rate, loss_function, iteration,
                snowflake_table_pred_name, pred_table_attributes,
                bucket, prefix, csv_name):

    df, X, Y = data_prep(snowflake_table_train_name, 
                         train_table_attributes, null_threshold)
    
    clf, categorical_feature_indicies = train_model(X, Y, test_proportion,
                                                    bootstrap_type, depth, 
                                                    learning_rate, loss_function,
                                                    iteration)

    csv_files = model_prediction(snowflake_table_pred_name, pred_table_attributes, clf, X, bucket, prefix, csv_name)

    #csv files [(csv, filename), (csv, filename) ....]

    for file in csv_files:
        print(file[1])

    
