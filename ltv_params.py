#Parameters

#Data location
client = 'PROGRESSIVE'
snowflake_table_train = 'AUGUST_PIQ_LTV_JOIN'

snowflake_table_pred = 'AUGUST_PIQ_PREDICT'

#Initial Feature Selection
features = ['LTV, SUPPLY_VENDOR, USER_HOUR_OF_WEEK, REGION, BROWSER, DMA,PIQ_CHILD_PRESENT_AGE_GROUPS, \
            DWELLING, ETHNIC_GROUP,PIQ_GENDER_GROUPS,PIQ_AGE_GROUPS_PRESENT,PIQ_AGE_GROUPS,PIQ_CHILDREN_ESTIMATE, \
            ETHNICIQ_V2,MARITAL,EDUCATION,PROPERTY_TYPE,PIQ_INCOME_GROUPS,PIQ_ADULT_ESTIMATE,PIQ_SPANISH_SPEAKING, \
            PIQ_PEOPLE_IN_HH,PIQ_PRESENCE_OF_CHILDREN']

null_threshold = 0.5

#Catboost Modeling Selection
bootstrap_type='MVS'
depth=6
learning_rate=0.05
loss_function= 'RMSE'
iteration=1000#("Enter Number of Iternations")#100
test_proportion=0.3

#Data Formatting
csv_name = 'chalice_test_daids.csv'
bucket = 'tradedesk-uploads'
prefix='daids_user_scores/chalice_demo'


    



    
