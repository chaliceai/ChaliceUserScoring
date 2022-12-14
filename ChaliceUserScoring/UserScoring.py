import snowflake.connector
import boto3
import json
import multiprocessing as mp
import csv

def post_data(advertiser_id, scores_csv, segment_name, secret_key, **kwargs):
    from ChaliceAPIUsage.APIConnection import handle_payload
    datakey = bytes(secret_key, 'utf-8')
    print("Using process pool execution for POST... ")
    payload_list = generate_user_score_payloads(file=scores_csv, segment_name=segment_name,
                                 advertiser_id=advertiser_id, datakey=datakey, **kwargs)
    total = 0
    failed = 0
    p = mp.Pool(processes=20)
    for num_total, num_failed in p.imap(handle_payload, payload_list):
        total += num_total
        failed += num_failed
    p.close()
    p.join()
    return {'TotalIDs': total, 'FailedIDs': failed}


def generate_user_score_payloads(file, segment_name, advertiser_id, datakey, **kwargs):
    from ChaliceAPIUsage.Utils.BidListGenerator import get_header_indices
    if not kwargs.get('timeToLiveCol'):
        timeToLiveCol = 'time_to_live'
    if not kwargs.get('valueCol'):
        valueCol = 'value'
    if not kwargs.get('idCol'):
        idCol = 'id'
    if not kwargs.get('idTypeCol'):
        idTypeCol = 'id_type'
    if not kwargs.get('max_lines'):
        max_lines = 10000
    if not kwargs.get('sep'):
        sep = ','

    items = []
    payloads = []

    idx = 0
    header_cols = get_csv_header_stream(file)

    indices = get_header_indices([timeToLiveCol, valueCol, idCol, idTypeCol], header_cols)
    for ln in csv_read_stream(file):
        idx += 1
        if idx == max_lines:
            idx = 0
            payloads.append({'datakey': datakey, 'AdvertiserID': advertiser_id, 'Items': items})
            items = []
        id_type_index = indices[idTypeCol]
        id_index = indices[idCol]
        time_to_live_index = indices[timeToLiveCol]
        value_index = indices[valueCol]
        items.append({ln[id_type_index]: ln[id_index],
                      'Data': [{'Name': segment_name,
                                'TtlInMinutes': ln[time_to_live_index],
                                'BaseBidCPM': ln[value_index]}]})
    payloads.append({'datakey': datakey, 'AdvertiserID': advertiser_id, 'Items': items})
    return payloads


def get_csv_header_stream(file):
    encoding = 'utf-8'
    f = file.readline().decode(encoding)
    if f[0] == '\ufeff':
        f = f.lstrip('\ufeff')
    
    f = f.splitlines()
    reader = csv.reader(f)
    for ln in reader:
        return ln


def csv_read_stream(file):
    encoding = 'utf-8'
    f = file.read().decode(encoding).splitlines()
    reader = csv.reader(f)
    for ln in reader:
        yield ln


def set_snowflake_connection(snowflake_secret_name):
    try:
        client = boto3.client('secretsmanager')
        snowflake_credentials = client.get_secret_value(SecretId=snowflake_secret_name)
        snowflake_credentials = json.loads(snowflake_credentials['SecretString'])

        sf_conn = snowflake.connector.connect(
            user = snowflake_credentials['dbUser'],
            account = snowflake_credentials['dbAccount'],
            password = snowflake_credentials['dbPassword'],
            warehouse = 'DEV_WH'
        )

        sf_cur = sf_conn.cursor()

        return sf_conn, sf_cur

    except Exception as err:
        print(err)


class UserScoring:

    
    def __init__(self, train_table_name=None, train_table_attributes=None, 
                 prediction_table_name=None, prediction_table_attributes=None,
                 null_threshold=0.5, bootstrap_type='MVS', depth=6,
                 loss_function='RMSE', iteration=1000, test_proportion=0.3,
                 learning_rate=0.05, s3_bucket=None, s3_prefix=None, 
                 csv_file_name=None):

        # Instance Variables
        # Data Location
        self._train_table_name = train_table_name
        self._train_table_attributes = train_table_attributes

        self._prediction_table_name = prediction_table_name
        self._prediction_table_attributes = prediction_table_attributes

        # Catboost Modeling
        self._null_threshold = null_threshold
        self._bootstrap_type = bootstrap_type
        self._depth = depth
        self._learning_rate = learning_rate
        self._loss_function = loss_function
        self._iteration = iteration
        self._test_proportion = test_proportion

        # Bucket Location
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix

        # Output File(s) name
        self._csv_file_name = csv_file_name

        # Check for parameters
        self._parameters_set = False

        # List for output file(s)
        self._csv_files_list = []


    def _is_parameters_set(self):
        if(self._parameters_set == True):
            return True

        is_parameters_set = True
        for param, value in self.__dict__.items():
            if(value == None):
                print(f"Please set the '{param}' attribute before running this function")
                print('Please run the set_params funtion to set this parameter')
                is_parameters_set = False

        self._parameters_set = is_parameters_set
        return is_parameters_set
        

    def set_params(self, train_table_name=None, train_table_attributes=None, 
                   prediction_table_name=None, prediction_table_attributes=None,
                   null_threshold=0.5, bootstrap_type='MVS', depth=6,
                   loss_function='RMSE', iteration=1000, test_proportion=0.3,
                   learning_rate=0.05, s3_bucket=None, s3_prefix=None, 
                   csv_file_name=None):
        
        if train_table_name != None:
            self._train_table_name = train_table_name
        
        if train_table_attributes != None:
            self._train_table_attributes = train_table_attributes
        
        if prediction_table_name != None:
            self._prediction_table_name = prediction_table_name
        
        if prediction_table_attributes != None:
            self._prediction_table_attributes = prediction_table_attributes

        if null_threshold != 0.5:
            self._null_threshold = null_threshold
        
        if bootstrap_type != 'MVS':
            self._bootstrap_type = bootstrap_type

        if depth != 6:
            self._depth = depth

        if loss_function != 'RSME':
            self._loss_function = loss_function

        if iteration != 1000:
            self._iteration = iteration

        if test_proportion != 0.3:
            self._test_proportion = test_proportion

        if learning_rate != 0.05:
            self._learning_rate = learning_rate

        if s3_bucket != None:
            self._s3_bucket = s3_bucket
        
        if s3_prefix != None:
            self._s3_prefix = s3_prefix

        if csv_file_name != None:
            self._csv_file_name = csv_file_name
       
    
    def run_user_scoring(self, snowflake_secret_name):
        from ChaliceUserScoring.Utils.run_ltv_pred import data_prep, train_model, model_prediction
        from ChaliceUserScoring.Utils.ltv_model import fix_nulls_and_types, log_transform
        from sklearn.model_selection import train_test_split
        from catboost import CatBoostRegressor, Pool, sum_models
        import shap 
        import numpy as np  
        import pandas as pd
        if not self._is_parameters_set():
            return

        sf_conn, sf_cur = set_snowflake_connection(snowflake_secret_name)

        attributes = ', '.join(self._train_table_attributes)
        sql =  f"SELECT {attributes} FROM {self._train_table_name} LIMIT 10000" #LIMIT ADDED FOR TESTING
        sf_cur.execute(sql)

        

        i = 0
        best_model = []
        prev_model = []
        while True:
            data = sf_cur.fetchmany(2000)
            print(f'Fetch ammount 2k')
            if not data:
                break
            
            df = pd.DataFrame(data, columns=self._train_table_attributes, copy=False)
            print('created data frame')
            print('length of df', len(df.columns))

            df = fix_nulls_and_types(df, self._null_threshold)
            print('Fixed Null and types')
            print('length of df', len(df.columns))

            X, Y = log_transform(df)
            # print('Log Transform')
            # print(X.columns)
            X_train, X_val, Y_train, Y_val = train_test_split(X,Y,test_size=self._test_proportion, random_state=0)

            # print('Length Of X_train', len(X_train.columns))
            # print('Length of X_val', len(X_val.columns))
            categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
        

            clf=CatBoostRegressor(iterations=self._iteration, 
                              depth=self._depth, 
                              learning_rate=self._learning_rate,
                              bagging_temperature = 0.5, 
                              custom_metric=['RMSE', 'R2'],
                              bootstrap_type=self._bootstrap_type,
                              l2_leaf_reg = 10,
                              loss_function=self._loss_function)

            if i == 0:
                train_pool = Pool(data=X_train, label=Y_train, cat_features=categorical_features_indices)
                test_pool = Pool(data=X_val, label=Y_val, cat_features=categorical_features_indices)
                
            else:
                train_pool = Pool(data=X_train, label=Y_train, cat_features=categorical_features_indices, baseline=sum_model.predict(X_train))
                test_pool = Pool(data=X_val, label=Y_val, cat_features=categorical_features_indices, baseline=sum_model.predict(X_val))

            clf.fit(train_pool,
                    eval_set=test_pool, plot=False, 
                    verbose=False)

            if i == 0:
                sum_model = clf
            
            sum_model = sum_models([clf, sum_model])

            print('\n\n')
            i += 1


        print('Last Model Ran ----')
        print('Best Iteration:')
        print(clf.get_best_iteration())
        res = clf.get_best_score()['learn']
        print(res)

        r2 = res['R2']
        rmse = res['RMSE']
        
        print('Train Metrics:')
        print('R2:',r2)
        print('RMSE:',rmse)
        
        res2 = clf.get_best_score()['validation']
        r2_2 = res2['R2']
        rmse_2 = res2['RMSE']
        
        print('Train Metrics:')
        print('R2:',r2_2)
        print('RMSE:',rmse_2)

        shap_values = clf.get_feature_importance(Pool(X_val, label=Y_val,cat_features=categorical_features_indices), 
                                                                         type="ShapValues")
        expected_value = shap_values[0,-1]
        shap_values = shap_values[:,:-1]

        shap.initjs()
        shap.force_plot(expected_value, shap_values[3,:], X_val.iloc[3,:])

        shap.summary_plot(shap_values, X_val)
        ##Feature importance
        clf.get_feature_importance(prettified=True)








        # csv_files = model_prediction(self._prediction_table_name, self._prediction_table_attributes, 
        #                            clf, X, self._s3_bucket, self._s3_prefix, self._csv_file_name, sf_cur)

        

        

        # Closing snowflake connection
        sf_conn.close()
        sf_cur.close()
        csv_files = []
        self._csv_files_list = csv_files

        for file in csv_files: 
            print(f"{file} was created and pushed to s3 Bucket: {self._s3_bucket}")
        
        return csv_files
    
    # TODO get each csv in s3 then push that. Only loading in one at a time.
    # This is to avoid having all of them in memory
    def push_to_TTD(self, csv_files_list, advertiser_id, segment_name, secret_key):
        import time
        import datetime as dt

        if (self._s3_bucket == None) or (self._s3_prefix == None):
            print('Please run setParams() and set the s3_bucket or s3_prefix variable')
            return

        failed_files = 0
        t1 = time.time()
        results = []
        num_files = len(csv_files_list)
        s3 = boto3.client('s3') 


        for i, csv_file_name in enumerate(csv_files_list):
            print(f'Uploading chunk {i + 1}/{num_files}')
           
            obj = s3.get_object(Bucket= self._s3_bucket, Key= f'{self._s3_prefix}/{csv_file_name}') 
            data = obj['Body']

            try:
                result = post_data(advertiser_id=advertiser_id,
                                    scores_csv=data,
                                    segment_name=segment_name,
                                    secret_key=secret_key)
                data.close()    
            except UnicodeDecodeError as e:
                print(f'**FAILED** Unable to read {csv_file_name}. Skipping...')
                failed_files += 1
                continue
            except Exception as e:
                print(f'**FAILED** Upload of {csv_file_name} failed due to {type(e)}. Skipping...')
                failed_files += 1
                raise e
            results.append(result)
        t2 = time.time()
        print(f'Success! Completed at: {dt.datetime.now()}. Time elapsed: {t2 - t1}')
        print('Total IDs Submitted:', sum([x["TotalIDs"] for x in results]))
        print('Total Lines with Errors:', sum([x['FailedIDs'] for x in results]))
        print(f'Number of files skipped due to error: {failed_files}')
