import numpy as np
import pandas as pd
from io import StringIO
from botocore.config import Config

# Formats the dataframe accordingly
def daid_format_pandas(daids):
    daids = daids[['DEVICE_ADVERTISING_ID','ltv']]
    daids = daids.rename(columns={'DEVICE_ADVERTISING_ID': 'id'})
    daids = daids.rename(columns={'ltv': 'value'})
    daids['id_type'] = 'DAID'
    daids['time_to_live'] = '43200'
    daids['value'] = np.exp(daids['value'])
    daids['value'] = daids['value'].mask(daids['value'] < 0, 0)
    return daids

# Pushes the dataframe to an s3 Bucket
def pandas_push_to_s3(df, bucket, prefix, file_name, aws_access, aws_secret):
    import boto3
    
    # Creating S3 Resource From the Session.
    s3_res = boto3.client('s3', aws_access_key_id=aws_access, aws_secret_access_key=aws_secret, region_name='us-east-1')
    csv_buffer = StringIO()
    prefix = f"{prefix}/{file_name}"
    df.to_csv(csv_buffer, index=False)
    s3_res.put_object(Body=csv_buffer.getvalue(), Bucket=bucket, Key=prefix)
    csv_buffer.close()
    del df
    print('File successfully placed in S3 Bucket')
    return (file_name)


    


    
