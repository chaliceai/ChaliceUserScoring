#Parameters

#Data location
client = 'PROGRESSIVE'
table_train_name = 'CLIENT.PROGRESSIVE.AUGUST_PIQ_JOIN_LTV_2'
table_train_attributes = ["LTV", "SUPPLY_VENDOR", "USER_HOUR_OF_WEEK", "REGION", "BROWSER", "DMA",
    "PIQ_CHILD_PRESENT_AGE_GROUPS", "DWELLING", "ETHNIC_GROUP", "PIQ_GENDER_GROUPS", "PIQ_AGE_GROUPS_PRESENT",
    "PIQ_AGE_GROUPS", "PIQ_CHILDREN_ESTIMATE", "ETHNICIQ_V2", "MARITAL", "EDUCATION", "PROPERTY_TYPE", "PIQ_INCOME_GROUPS",
    "PIQ_ADULT_ESTIMATE", "PIQ_SPANISH_SPEAKING", "PIQ_PEOPLE_IN_HH", "PIQ_PRESENCE_OF_CHILDREN"]


table_pred_name = 'CLIENT.PROGRESSIVE.PIQ_REDS_0811'
table_pred_attributes = ["DEVICE_ADVERTISING_ID", "SUPPLY_VENDOR", "USER_HOUR_OF_WEEK", "REGION", "BROWSER", "DMA",
    "PIQ_CHILD_PRESENT_AGE_GROUPS", "DWELLING", "ETHNIC_GROUP", "PIQ_GENDER_GROUPS", "PIQ_AGE_GROUPS_PRESENT",
    "PIQ_AGE_GROUPS", "PIQ_CHILDREN_ESTIMATE", "ETHNICIQ_V2", "MARITAL", "EDUCATION", "PROPERTY_TYPE", "PIQ_INCOME_GROUPS",
    "PIQ_ADULT_ESTIMATE", "PIQ_SPANISH_SPEAKING", "PIQ_PEOPLE_IN_HH", "PIQ_PRESENCE_OF_CHILDREN "]

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

#Secrets/other
ADVERTISER_ID = 'ca5g5oz'
ttd_user = "ttd_api_hregxvd@chalice.com"

NUM_FILES = 1

    



    
