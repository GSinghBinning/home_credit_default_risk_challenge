import src.data.preprocess as pp
import pandas as pd
import pytest
import datatest as dt


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_test():
    df = pd.read_csv("../datafiles_for_tests/testing_set_test.csv")
    return df


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_train():
    df = pd.read_csv("../datafiles_for_tests/testing_set_train.csv")
    return df


def test_encode_categorical_dtypes(df_train, df_test):
    df_train, df_test = df_train.align(df_test, join='inner', axis=1)
    df_train_result, df_test_result = pp.encode_categorical(df_train, df_test)

    for column in df_train_result:
        assert df_train_result[column].dtype != "object", \
            "Training set has still object column after categorical encoder"
        assert df_test_result[column].dtype != "object", \
            "Test set has still object column after categorical encoder"


def test_encode_categorical_dtypes_shape(df_train, df_test):
    df_train, df_test = df_train.align(df_test, join='inner', axis=1)

    df_train_result, df_test_result = pp.encode_categorical(df_train, df_test)

    assert df_train_result.shape == df_test_result.shape, "Shape of the dataframes is not aligned"


def test_data_cleaning_application_values(df_train):
    df_train = pp.data_cleaning_application(df_train)

    assert df_train['CODE_GENDER'].nunique() == 2, \
        "Code Gender still has not 2 unique values"
    assert len(df_train[df_train['DAYS_EMPLOYED'] == 365243]) == 0, \
        "There are still some 365243 values in DAYS EMPLOYED"

""" wannn value wann assertion"""