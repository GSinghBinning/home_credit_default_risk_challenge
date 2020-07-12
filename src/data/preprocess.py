import pandas as pd
from sklearn.preprocessing import LabelEncoder


def one_hot_encoder(df):
    df = pd.get_dummies(df)
    return df


def label_encoder(df):
    l_encoder = LabelEncoder()
    l_encoder.fit_transform(df)
    return df


def encode_categorical(df):
    l_encoder_count = 0
    o_h_encoder_count = 0
    for column in df:
        if df[column].dtype == "object":

            if len(list(df[column].dtype.unique())) <= 2:
                df[column] = label_encoder(df[column])
                l_encoder_count += 1
            elif len(list(df[column].dtype.unique())) > 2:
                df[column] = one_hot_encoder(df[column])
                o_h_encoder_count += 1

    print("Total of %s columns transformed with Label Encoder" % l_encoder_count)
    print("Total of %s columns transformed with One Hot Encoder" % o_h_encoder_count)
