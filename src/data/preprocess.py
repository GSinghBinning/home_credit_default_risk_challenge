import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


def one_hot_encoder(df):
    df = pd.get_dummies(df)
    return df


def label_encoder(df):
    l_encoder = LabelEncoder()
    df = l_encoder.fit_transform(df)
    return df


def encode_categorical(df_train, df_test):
    l_encoder_count = 0
    o_h_encoder_count = 0
    for column in df_train:
        if df_train[column].dtype == "object":

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
    df = df[df['CODE_GENDER'] != 'XNA']
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    print("outlier and faulty values has been processed")
    return df


def impute(df_train, df_test):
    imputer = SimpleImputer(strategy='median')
    imputer.fit(df_train)
    df_train = imputer.transform(df_train)
    df_test = imputer.transform(df_test)
    print("NAN values imputed with Simple Imputer - strategy = median")
    return df_train, df_test


def scale(df_train, df_test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_train)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)
    print("MinMaxScaler transformed the data")
    return df_train, df_test
