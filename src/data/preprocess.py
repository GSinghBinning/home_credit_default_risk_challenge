import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


def one_hot_encoder(df):
    """ Encodes columns of a df with categorical values, which have more than 2 unique values"""
    df = pd.get_dummies(df)
    return df


def label_encoder(df):
    """Encodes columns of a df with categorical values, which have 2 or less unique values"""
    l_encoder = LabelEncoder()
    df = l_encoder.fit_transform(df)
    return df


def encode_categorical(df_train, df_test):
    """Gets a df with the train set and one df with the test set,
     and decides for every column, if the column should be encoded with the
      label encoder or one hot encoder, based on the number of unique values"""

    # Counter values to print out ne number of encoded columns
    l_encoder_count = 0
    o_h_encoder_count = 0
    # for loop to iterate through every column of df_train and apply the encoding to test
    # and train df, where the data type of the values is object
    for column in df_train:
        if df_train[column].dtype == "object":
            # Check if the unique categorical values max 2 or more
            if len(list(df_train[column].unique())) <= 2:
                df_train[column] = label_encoder(df_train[column])
                df_test[column] = label_encoder(df_test[column])
                l_encoder_count += 1
            elif len(list(df_train[column].unique())) > 2:
                df_train[column] = one_hot_encoder(df_train[column])
                df_test[column] = one_hot_encoder(df_test[column])
                o_h_encoder_count += 1

    print("Total of %s columns transformed with Label Encoder" % l_encoder_count)
    print("Total of %s columns transformed with One Hot Encoder" % o_h_encoder_count)

    return df_train, df_test


def data_cleaning_application(df):
    """ This function deletes the rows with faulty CODE_GENDER values and
    the outliers in DAYS_EMPLOYED by replacing it with NAN values"""

    df = df[df['CODE_GENDER'] != 'XNA'].reset_index(drop=True)
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    print("outlier and faulty values has been processed")
    return df


def impute(df_train, df_test):
    """ This function imputes nan values in the train and test set with
     the SimpleImputer from the sklearn package"""

    imputer = SimpleImputer(strategy='median')

    # Fitting process with the train df
    imputer.fit(df_train)

    # Transforming both dataframes with fitted imputer
    df_train = imputer.transform(df_train)
    df_test = imputer.transform(df_test)
    print("NAN values imputed with Simple Imputer - strategy = median")
    return df_train, df_test


def scale(df_train, df_test):
    """This function scales the values of the received test and train dataframes
    with the MinMaxScaler  and feature_range 0 and 1"""

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fitting process with the train df
    scaler.fit(df_train)

    # Transforming both dataframes with fitted scaler
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)
    print("MinMaxScaler transformed the data")
    return df_train, df_test
