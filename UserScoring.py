import snowflake.connector
from Utils.ltv_model import *


class UserScoring():
    def __init__(self):
        
        # Data Location
        self.train_table_name = None
        self.train_table_attributes = None

        self.prediction_table_name = None
        self.train_table_attributes = None

        self.null_treshhold = None

        # Catboost Modeling
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

        # Connection to snowflake
        self._sf_conn = snowflake.connector.connect(
            user='TUCKER_DEV',
            password='RemoteAccess1',
            account='mja29153.us-east-1',
        )
        self._ctx = self._sf_conn.cursor()

        # Check for parameters
        self.parameters_set = None

    def isParametersSet(self):

        if(self.parameters_set == True):
            return True

        is_parameters_set = True
        for param, value in self.__dict__.items():
            if(value == None):
               print(f"Please set the '{param}' parameter before running this function")
               is_parameters_set = False

        self.parameters_set = is_parameters_set
        return is_parameters_set
        
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
        

        
        
                


a = UserScoring()
a.isParametersSet()