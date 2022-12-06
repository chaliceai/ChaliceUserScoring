# Chalice User Scoring Package

## Table of contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [How to Use](#how-to-use)
    - [constructor](#constructor)
    - [set_params()](#set_params)
    - [run_user_scoring()](#run_user_scoring)
    - [push_to_TTD()](#push_to_ttd)


## Introduction
This package will run the user scoring models based on the input provided.


## Getting Started
This getting started assumes you have the aws cli installed with the credentials file set and
have permissions from Chalice AI to access the repo.

Use the following command to install the package using `pip`

```
 pip install git+https://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@github.com/chaliceai/ChaliceUserScoring.git@feature-python-packaging
```
or using SSH
```
pip install --no-input git+ssh://git@github.com/chaliceai/ChaliceUserScoring.git@feature-python-packaging
```

This will install the `Chalice User Scoring` package into your python environment. Use of a
virtual environment is recommended.

Importing Tip:
```py
from ChaliceUserScoring.UserScoring import UserScoring
```


## How To Use
**Methods**
- [constructor](#constructor)
- [set_params()](#set_params)
- [get_user_scoring()](#get_user_scoring)
- [push_to_TTD()](#push_to_ttd)


### constructor
```py
UserScoring.__init__(train_table_name, train_table_attributes, 
                     prediction_table_name, prediction_table_attributes,
                     null_threshold, bootstrap_type, depth,
                     loss_function, iteration, test_proportion,
                     learning_rate, s3_bucket, s3_prefix, 
                     csv_file_name):
```
Parameters:
| Parameter                 | Type              | Default|
| -------------             |:-------------:    | :-----:|
| train_table_name          | `string`          | *None* |
| train_table_attributes    | `list of strings` |  *None* |
| prediction_table_name     | `string`          |  *None* |
| null_threshold            | `float`           | *0.5* |
| bootstrap_type            | `string`          | *'MVS'* |
| depth                     | `int`             | *6* |
| loss_function             | `string`          | *'RMSE'* |
| iteration                 | `int`             | 1000 |
| test_proportion           | `float`           | *0.3* |
| learning_rate             | `float`           | *0.05* |
| s3_bucket                 | `string`          | *None* |
| s3_prefix                 | `string`          | *None* |
| csv_file_name             | `string`          | *None* |

Example:
```py
my_user_scoring = UserScoring(train_table_name='TABLE_NAME', train_table_attributes=['Attribute', 'Attribute'],
                              s3_bucket='BUCKET_NAME')                        
 ```
 
 or
 
 ```py
 my_user_scoring = UserScoring()
 ```
 
[Back to Table of Contents](#table-of-contents)

### set_params()

```py
UserScoring.set_params(train_table_name, train_table_attributes, 
                       prediction_table_name, prediction_table_attributes,
                       null_threshold, bootstrap_type, depth,
                       loss_function, iteration, test_proportion,
                       learning_rate, s3_bucket, s3_prefix, 
                       csv_file_name)
```

Parameters:
| Parameter                 | Type              | Default|
| -------------             |:-------------:    | :-----:|
| train_table_name          | `string`          | *None* |
| train_table_attributes    | `list of strings` |  *None* |
| prediction_table_name     | `string`          |  *None* |
| null_threshold            | `float`           | *0.5* |
| bootstrap_type            | `string`          | *'MVS'* |
| depth                     | `int`             | *6* |
| loss_function             | `string`          | *'RMSE'* |
| iteration                 | `int`             | 1000 |
| test_proportion           | `float`           | *0.3* |
| learning_rate             | `float`           | *0.05* |
| s3_bucket                 | `string`          | *None* |
| s3_prefix                 | `string`          | *None* |
| csv_file_name             | `string`          | *None* |

Example:
```py
my_user_scoring = UserScoring()

my_user_scoring.set_params(prediction_table_name='TABLE_NAME', depth=10, csv_file_name='FILE_NAME')
```
[Back to Table of Contents](#table-of-contents)

### run_user_scoring()
```py
  UserScoring.run_user_scoring(snowflake_secret_name):
 ```
 Parameters:
| Parameter                 | Type              | Default|
| -------------             |:-------------:    | :-----:|
| snowflake_secret_name     | `string`          | *None* |

Returns:
`List of Strings` containing csv file names pushed to s3

Example:
```py
my_user_scoring = UserScoring()

csv_files = my_user_scoring.run_user_scoring()
```

[Back to Table of Contents](#table-of-contents)

### push_to_TTD()
```py
UserScoring.push_to_TTD(ttd_user, advertiser_id, segment_name, secret_key):
```
| Parameter                 | Type              | Default|
| -------------             |:-------------:    | :-----:|
| ttd_user                  | `string`          | *None* |
| advertiser_id             | `string`          | *None* |
| segment_name              | `string`          | *None* |
| secret_key                | `string`          | *None* |

Example:
```py
my_user_scoring = UserScoring()
csv_files_list = my_user_scoring.run_user_scoring()
my_user_scoring.push_to_TTD(csv_files_list, 'ADVERTISER_ID', 'SEGMENT_NAME', 'SECRET_KEY')
```
or
```py
csv_files_list = ['csv_file_name1.csv' ,'csv_file_name2.csv]
my_user_scoring.push_to_TTD(csv_files_list, 'ADVERTISER_ID', 'SEGMENT_NAME', 'SECRET_KEY')
```
[Back to Table of Contents](#table-of-contents)

