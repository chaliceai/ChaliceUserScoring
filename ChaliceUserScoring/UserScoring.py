import snowflake.connector
import boto3
import json

class UserScoring:

    def __del__(self):
        self.sf_conn.close()
        self.sf_cur.close()

    
    def __init__(self):

        # Instance Variables
        # Data Location
        self.train_table_name = None
        self.train_table_attributes = None

        self.prediction_table_name = None
        self.prediction_table_attributes = None

        # Catboost Modeling
        self.null_threshold = None
        self.bootstrap_type = None
        self.depth = None
        self.learning_rate = None
        self.loss_function = None
        self.iteration = None
        self.test_proportion = None

        # Bucket Location
        self.s3_bucket = None
        self.s3_prefix = None

        # Output File(s) name
        self.csv_file_name = None

        # Check for parameters
        self.parameters_set = False

        # list for output file(s)
        self.csv_files_list = []

    def setSnowflakeConnection(self):

        try:
            client = boto3.client('secretsmanager')
            snowflake_credentials = client.get_secret_value(SecretId='chalice-dev-config-snowflake-credentials') # Hardcoded Secret name
            snowflake_credentials = json.loads(snowflake_credentials['SecretString'])

            self.sf_conn = snowflake.connector.connect(
                user = snowflake_credentials['dbUser'],
                account = snowflake_credentials['dbAccount'],
                password = snowflake_credentials['dbPassword'],
                warehouse = 'COMPUTE_WH'
            )

            self.sf_cur = self.sf_conn.cursor()
            print('Sucessfully connected to snowflake')

        except Exception as err:
            print(err)

        


    def isParametersSet(self):
        if(self.parameters_set == True):
            return True

        is_parameters_set = True
        for param, value in self.__dict__.items():
            if(value == None):
               print(f"Please set the '{param}' attribute before running this function")
               is_parameters_set = False

        self.parameters_set = is_parameters_set
        return is_parameters_set
        
    def runUserScoring(self):
        from ChaliceUserScoring.Utils.run_ltv_pred import data_prep, train_model, model_prediction
        
        if not self.isParametersSet():
            return
       
        df, X, Y = data_prep(self.sf_cur, self.train_table_name, 
                             self.train_table_attributes, self.null_threshold)
        
        clf, categorical_feature_indicies = train_model(X, Y, self.test_proportion,
                                                        self.bootstrap_type, self.depth, 
                                                        self.learning_rate, self.loss_function,
                                                        self.iteration)

        csv_files = model_prediction(self.prediction_table_name, self.prediction_table_attributes, 
                                     clf, X, self.s3_bucket, self.s3_prefix, self.csv_file_name, self.sf_cur)

        self.csv_files_list = csv_files

        #csv files [(csv, filename), (csv, filename) ....]
        for file in csv_files: 
            print(f"{file[1]} was created and pushed to s3 Bucket: {self.s3_bucket}")
    
  
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
        num_files = len(self.csv_files_list)
        for i, csv in enumerate(self.csv_files_list):
            print(f'Uploading chunk {i + 1}/{num_files}')
            print(csv[1].format(i + 1))
            try:
                result = conn.post_data(advertiser_id=advertiser_id,
                                        scores_csv=csv[0].format(i + 1),
                                        segment_name=segment_name)
            except UnicodeDecodeError as e:
                print(f'**FAILED** Unable to read {csv[1].format(i + 1)}. Skipping...')
                failed_files += 1
                continue
            except Exception as e:
                print(f'**FAILED** Upload of {csv[1].format(i + 1)} failed due to {type(e)}. Skipping...')
                failed_files += 1
                raise e
            results.append(result)
        t2 = time.time()
        print(f'Success! Completed at: {dt.datetime.now()}. Time elapsed: {t2 - t1}')
        print('Total IDs Submitted:', sum([x["TotalIDs"] for x in results]))
        print('Total Lines with Errors:', sum([x['FailedIDs'] for x in results]))
        print(f'Number of files skipped due to error: {failed_files}')

