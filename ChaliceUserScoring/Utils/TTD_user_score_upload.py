import numpy as np
import pandas as pd
from io import StringIO

#reads csv, formats it, creates new csv.
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

def daid_format_pandas(daids):
    daids = daids[['DEVICE_ADVERTISING_ID','ltv']]
    daids = daids.rename(columns={'DEVICE_ADVERTISING_ID': 'id'})
    daids = daids.rename(columns={'ltv': 'value'})
    daids['id_type'] = 'DAID'
    daids['time_to_live'] = '43200'
    daids['value'] = np.exp(daids['value'])
    daids['value'] = daids['value'].mask(daids['value'] < 0, 0)
    return daids

def pandas_push_to_s3(df, bucket, prefix, file_name):
    import boto3

    # Creating S3 Resource From the Session.
    s3_res = boto3.resource('s3')
    csv_buffer = StringIO()
    prefix = f"{prefix}/{file_name}"
    df.to_csv(csv_buffer)
    s3_res.Object(bucket, prefix).put(Body=csv_buffer.getvalue())

    print('File successfully placed in S3 Bucket')
    return (df, file_name)


#Pushes csv to s3
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
    


    
