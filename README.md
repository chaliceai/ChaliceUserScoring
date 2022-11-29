# Chalice User Scoring Package

## Table of contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [How to Use](#how-to-use)
    - [constructor](#constructor)
    - [set_snowflake_credentials()](#set_snowflake_credentials())
    - [set_params()](#set_params())
    - [get_user_scoring()](#get_user_scoring())


## Introduction
This package will run the user scoring models based on the input provided.


## Getting Started
This getting started assumes you have the aws cli installed with the credentials file set and
have permissions by Chalice AI to access the repo.

Use the following command to install the package using `pip`

> pip install git@something

This will install the `Chalice User Scoring` package into your python environment. Use of a
virtual environment is recommended.


## How To Use
** Methods **
    - [constructor](#constructor)
    - [set_snowflake_connection()](#set_snowflake_connection())
    - [set_params()](#set_params())
    - [get_user_scoring()](#get_user_scoring())

### constructor


### set_snowflake_credentials(snowflake_credentials_secret=*string*) {#set_snowflake_connection()}
This function uses the `aws secrets manager` to fetch the username and password in order to instantiate 
a snowflake conncetion and cursor.

### set_params()

### get_user_scoring()




