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
        's3_prefix': 'HenryOnboarding/DatabricksTest',
        
        # AWS secrets
        'aws_access': 'AKIA2XH7LDGPOVEIQWUT',
        'aws_secret': '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',
    }

    # Secrets/other
    segment_name = "TEST_user_scoring_AIRFLOW"
    user_scoring_secret = "CHALICE_TTD_USER_SCORING_CREDENTIALS"

    snowflake_secret_name = 'chalice-dev-config-snowflake-credentials'

    my_user_scoring = UserScoring(**params)
  
    csv_list = my_user_scoring.run_user_scoring(snowflake_secret_name)

    my_user_scoring.push_to_TTD(csv_list, segment_name, user_scoring_secret)

if __name__ == '__main__':
    test()