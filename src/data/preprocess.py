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


def encode_categorical(df):
    l_encoder_count = 0
    o_h_encoder_count = 0
    for column in df:
        if df[column].dtype == "object":

            if len(list(df[column].unique())) <= 2:
                df[column] = label_encoder(df[column])
                l_encoder_count += 1
            elif len(list(df[column].unique())) > 2:
                df[column] = one_hot_encoder(df[column])
                o_h_encoder_count += 1

    print("Total of %s columns transformed with Label Encoder" % l_encoder_count)
    print("Total of %s columns transformed with One Hot Encoder" % o_h_encoder_count)


def data_cleaning_application(df):
    df = df[df['CODE_GENDER'] != 'XNA']
    df['DAYS_EMPLOYED'].replace(365423, np.nan, inplace=True)


"""
WICHTIG : ERST DIE TARGET NOCH DROPPEN UND DIE DATEIEN KOPIEREN



SONST IMPUTET ER DIE TARGETS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

def impute_nan_values(df):
    imputer = SimpleImputer(strategy = 'median')
    imputer.fit(df)
    df = imputer.transform(df)

def scale_minmax(df):
    scaler = MinMaxScaler(feature_range = (0,1))
    scaler.fit(df)
    df = scaler.transform(df)