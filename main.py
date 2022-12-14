from ChaliceUserScoring.UserScoring import UserScoring
import json
def test():
    # Parameters

    # Data location
    client = 'PROGRESSIVE'
    params = {
        'train_table_name': 'CLIENT.PROGRESSIVE.AUGUST_PIQ_JOIN_LTV_2',

        'train_table_attributes': ["LTV", "SUPPLY_VENDOR", "USER_HOUR_OF_WEEK", "REGION", "BROWSER", "DMA",
            "PIQ_CHILD_PRESENT_AGE_GROUPS", "DWELLING", "ETHNIC_GROUP", "PIQ_GENDER_GROUPS", "PIQ_AGE_GROUPS_PRESENT",
            "PIQ_AGE_GROUPS", "PIQ_CHILDREN_ESTIMATE", "ETHNICIQ_V2", "MARITAL", "EDUCATION", "PROPERTY_TYPE", "PIQ_INCOME_GROUPS",
            "PIQ_ADULT_ESTIMATE", "PIQ_SPANISH_SPEAKING", "PIQ_PEOPLE_IN_HH", "PIQ_PRESENCE_OF_CHILDREN"],


        'prediction_table_name': 'CLIENT.PROGRESSIVE.PIQ_REDS_0811',

        'prediction_table_attributes': ["DEVICE_ADVERTISING_ID", "SUPPLY_VENDOR", "USER_HOUR_OF_WEEK", "REGION", "BROWSER", "DMA",
            "PIQ_CHILD_PRESENT_AGE_GROUPS", "DWELLING", "ETHNIC_GROUP", "PIQ_GENDER_GROUPS", "PIQ_AGE_GROUPS_PRESENT",
            "PIQ_AGE_GROUPS", "PIQ_CHILDREN_ESTIMATE", "ETHNICIQ_V2", "MARITAL", "EDUCATION", "PROPERTY_TYPE", "PIQ_INCOME_GROUPS",
            "PIQ_ADULT_ESTIMATE", "PIQ_SPANISH_SPEAKING", "PIQ_PEOPLE_IN_HH", "PIQ_PRESENCE_OF_CHILDREN "],

        # Data Formatting
        'csv_file_name': 'TEST-python-module-User-Scoring.csv',

        's3_bucket': 'chaliceai-sandbox',
        's3_prefix': 'HenryOnboarding/ChaliceUserScoringTEST'
    }

    # Secrets/other
    ADVERTISER_ID = 'ca5g5oz'
    ttd_user = "ttd_api_hregxvd@chalice.com"
    secret_key = 'uar6lmos5mfccyjb26yhcepab3bmccel'
    segment_name = "TEST"

    snowflake_secret_name = 'chalice-dev-config-snowflake-credentials'

    my_user_scoring = UserScoring(**params)
  
    my_user_scoring.run_user_scoring(snowflake_secret_name)

    #test_files = ['testcsv.csv', 'utf_encoding_test_2.csv', 'TEST-python-module-User-Scoring_1_2022-12-02.csv']
    test_files = ['TEST-python-module-User-Scoring_1_2022-12-09.csv', 
                  'TEST-python-module-User-Scoring_2_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_3_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_4_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_5_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_6_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_7_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_8_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_9_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_10_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_11_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_12_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_13_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_14_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_15_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_16_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_17_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_18_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_19_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_20_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_21_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_22_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_23_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_24_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_25_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_26_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_27_2022-12-09.csv',
                  'TEST-python-module-User-Scoring_28_2022-12-09.csv']
   # my_user_scoring.push_to_TTD(test_files, ADVERTISER_ID, "TEST_user_scoring_AIRFLOW", secret_key)

if __name__ == '__main__':
    test()