from ChaliceAPIUsage.APIConnection import *
from Catboost_SMOTE import *
from ltv_model import *
from TTD_user_score_upload import *
from ltv_params import *

import warnings
warnings.filterwarnings("ignore")

def data_prep(snowflake_table_train, null_threshold):
    df = training_data(snowflake_table_train)
    dataframe = fix_nulls_and_types(df, nulls = 'Yes', types = 'Yes', val = 0, null_threshold = null_threshold)
    X, Y= log_transform(dataframe)
    
    return df, X, Y

def train_model(X,Y,test_proportion):
    X_train, X_val, Y_train, Y_val = testtrainsplit(X,Y,test_proportion=test_proportion)
    clf, categorical_features_indices = catboost_regression(X_train, X_val, Y_train, Y_val, bootstrap_type=bootstrap_type ,depth=depth,learning_rate=learning_rate,loss_function=loss_function, iteration=iteration)
    return clf, categorical_features_indices

def model_prediction(snowflake_table_pred, clf, X):
    df_piq = prediction_data(snowflake_table_pred)
    df_piq = catboost_prediction(df_piq, clf, X)
    return df_piq

def format_results(csv_name):
    daids = daid_format(csv_name = csv_name)
    push_to_s3(csv_name=csv_name, bucket=bucket, prefix=prefix)
    return daids

NUM_FILES = 1
def push_to_TTD(NUM_FILES=1, USER="ttd_api_hregxvd@chalice.com", ADVERTISER_ID = 'ca5g5oz'):
    from ChaliceAPIUsage.APIConnection import TradedeskAPIConnection
    conn = TradedeskAPIConnection("ttd_api_hregxvd@chalice.com")
    conn.set_secret(advertiser_id='ca5g5oz',
                    secret_key='uar6lmos5mfccyjb26yhcepab3bmccel')

    fname = '1014_daids_test_upload.csv'#'s3://tradedesk-uploads/daids_user_scores/chalice_demo/chalice_test_daids.csv'

    failed_files = 0
    t1 = time.time()
    results = []
    for i in range(NUM_FILES):
        print(f'Uploading chunk {i + 1}/{NUM_FILES}')
        print(fname.format(i + 1))
        try:
            result = conn.post_data(advertiser_id='ca5g5oz',
                                scores_csv=fname.format(i + 1),
                                segment_name='LTVUploadOctober')
        except UnicodeDecodeError as e:
            print(f'**FAILED** Unable to read {fname.format(i + 1)}. Skipping...')
            failed_files += 1
            continue
        except Exception as e:
            print(f'**FAILED** Upload of {fname.format(i + 1)} failed due to {type(e)}. Skipping...')
            failed_files += 1
            raise e
        results.append(result)
    t2 = time.time()
    print(f'Success! Completed at: {dt.datetime.now()}. Time elapsed: {t2 - t1}')
    print('Total IDs Submitted:', sum([x["TotalIDs"] for x in results]))
    print('Total Lines with Errors:', sum([x['FailedIDs'] for x in results]))
    print(f'Number of files skipped due to error: {failed_files}')
    

    
