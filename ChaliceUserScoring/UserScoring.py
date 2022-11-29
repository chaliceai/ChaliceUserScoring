import snowflake.connector
import boto3
import json

class UserScoring:


    def __del__(self):
        self._sf_conn.close()
        self._sf_cur.close()

    
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


    def set_snowflake_connection(self, snowflake_secret_name):
        try:
            client = boto3.client('secretsmanager')
            snowflake_credentials = client.get_secret_value(SecretId=snowflake_secret_name)
            snowflake_credentials = json.loads(snowflake_credentials['SecretString'])

            self._sf_conn = snowflake.connector.connect(
                user = snowflake_credentials['dbUser'],
                account = snowflake_credentials['dbAccount'],
                password = snowflake_credentials['dbPassword'],
                warehouse = 'COMPUTE_WH'
            )

            self._sf_cur = self._sf_conn.cursor()
            print('Sucessfully connected to snowflake')

        except Exception as err:
            print(err)


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
       
       
    def run_user_scoring(self):
        from ChaliceUserScoring.Utils.run_ltv_pred import data_prep, train_model, model_prediction
        
        if not self._is_parameters_set():
            return
       
        df, X, Y = data_prep(self._sf_cur, self._train_table_name, 
                             self._train_table_attributes, self._null_threshold)

        clf, categorical_feature_indicies = train_model(X, Y, self._test_proportion,
                                                        self._bootstrap_type, self._depth, 
                                                        self._learning_rate, self._loss_function,
                                                        self._iteration)

        csv_files = model_prediction(self._prediction_table_name, self._prediction_table_attributes, 
                                     clf, X, self._s3_bucket, self._s3_prefix, self._csv_file_name, self._sf_cur)

        self._csv_files_list = csv_files

        #csv files [(csv, filename), (csv, filename) ....]
        for file in csv_files: 
            print(f"{file[1]} was created and pushed to s3 Bucket: {self._s3_bucket}")
    
  
    def push_to_TTD(self, user, advertiser_id, segment_name, secret_key):
        from ChaliceAPIUsage.APIConnection import TradedeskAPIConnection
        import time
        import datetime as dt

        conn = TradedeskAPIConnection(user)
        conn.set_secret(advertiser_id=advertiser_id,
                        secret_key=secret_key)

        failed_files = 0
        t1 = time.time()
        results = []
        num_files = len(self._csv_files_list)
        for i, csv in enumerate(self._csv_files_list):
            print(f'Uploading chunk {i + 1}/{num_files}')
            print(csv[1])
            csv[0].to_csv(path_or_buf=csv[1], index=False)
            try:
                result = conn.post_data(advertiser_id=advertiser_id,
                                        scores_csv=csv[1],
                                        segment_name=segment_name)
            except UnicodeDecodeError as e:
                print(f'**FAILED** Unable to read {csv[1]}. Skipping...')
                failed_files += 1
                continue
            except Exception as e:
                print(f'**FAILED** Upload of {csv[1]} failed due to {type(e)}. Skipping...')
                failed_files += 1
                raise e
            results.append(result)
        t2 = time.time()
        print(f'Success! Completed at: {dt.datetime.now()}. Time elapsed: {t2 - t1}')
        print('Total IDs Submitted:', sum([x["TotalIDs"] for x in results]))
        print('Total Lines with Errors:', sum([x['FailedIDs'] for x in results]))
        print(f'Number of files skipped due to error: {failed_files}')

