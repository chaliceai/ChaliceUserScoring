import numpy as np
import pandas as pd
from io import StringIO

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
def pandas_push_to_s3(df, bucket, prefix, file_name):
    import boto3

    # Creating S3 Resource From the Session.
    s3_res = boto3.resource('s3')
    csv_buffer = StringIO()
    prefix = f"{prefix}/{file_name}"
    df.to_csv(csv_buffer)
    s3_res.Object(bucket, prefix).put(Body=csv_buffer.getvalue())
    csv_buffer.close()
    del df
    print('File successfully placed in S3 Bucket')
    return (file_name)


    


    
