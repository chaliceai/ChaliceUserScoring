from ChaliceAPIUsage.APIConnection import *
import time
import datetime as dt
import numpy as np

def daid_format(csv_name):
    daids = pd.read_csv(csv_name, low_memory=False)
    daids = daids[['DEVICE_ADVERTISING_ID','ltv']]
    daids = daids.rename(columns={'DEVICE_ADVERTISING_ID': 'id'})
    daids = daids.rename(columns={'ltv': 'value'})
    daids['id_type'] = 'DAID'
    daids['time_to_live'] = '43200'
    daids['value'] = np.exp(daids['value'])
    daids['value'] = daids['value'].mask(daids['value'] < 0, 0)
    daids.to_csv('1014_daids_test_upload.csv', index = False)
    return daids
def push_to_s3(csv_name, bucket, prefix):
    import boto3
#Creating Session With Boto3.
    session = boto3.Session(
    aws_access_key_id='AKIA2XH7LDGPOVEIQWUT',
    aws_secret_access_key='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'
    )
    chalice_s3 = boto3.resource('s3', aws_access_key_id='AKIA2XH7LDGPOVEIQWUT',
    aws_secret_access_key='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')
    
    import os
    for i in range(1):
        bucket = 'tradedesk-uploads'
        prefix = f'daids_user_scores/chalice_demo'
        prediction_path = os.path.join(prefix, f'1014_daids_test_upload.csv')
        response = chalice_s3.Bucket(bucket).Object(prediction_path).upload_file(f'1014_daids_test_upload.csv')
    


    
