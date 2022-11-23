import snowflake.connector

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
        
        

        
        
                


a = UserScoring()
a.isParametersSet()