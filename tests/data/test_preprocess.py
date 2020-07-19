import src.data.preprocess as pp
import pandas as pd
import pytest
import datatest as dt
import numpy as np


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


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_train_encoded():
    df = pd.read_csv('../datafiles_for_tests/testing_set_train_encoded.csv')
    return df


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_test_encoded():
    df = pd.read_csv('../datafiles_for_tests/testing_set_test_encoded.csv')
    return df


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_test_imputed():
    test_array = np.loadtxt('../datafiles_for_tests/testing_set_test_imputed.csv', delimiter=',')
    return test_array


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_train_imputed():
    train_array = np.loadtxt('../datafiles_for_tests/testing_set_train_imputed.csv', delimiter=',')
    return train_array


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

    assert df_train.CODE_GENDER.nunique() == 2, \
        "The amount of unique CODE_GENDER doesn't match the number 2 "
    assert (df_train['DAYS_EMPLOYED'] == 365243).sum() == 0, \
        "There are still some faulty values(365243) in DAYS_EMPLOYED"


def test_data_cleaning_application_shape(df_train):
    df_train = pp.data_cleaning_application(df_train)

    assert df_train.shape == (999, 122), \
        "The resulting shape of df doesn't match the expected"


def test_impute_shape(df_train_encoded, df_test_encoded):
    df_train_imputed, df_test_imputed = pp.impute(df_train_encoded, df_test_encoded)

    assert df_train_encoded.shape == df_train_imputed.shape, \
        "Imputing changed the shape of train dataframe"
    assert df_test_encoded.shape == df_test_imputed.shape, \
        "Imputing changed the shape of test dataframe"


def test_impute_values(df_train_encoded, df_test_encoded):
    df_train_imputed, df_test_imputed = pp.impute(df_train_encoded, df_test_encoded)

    assert np.isnan(np.sum(df_train_imputed)) == False, \
        "there are still some NAN values in the train dataframe"
    assert np.isnan(np.sum(df_test_imputed)) == False, \
        "there are still some NAN values in the test dataframe"


def test_impute_dtypes(df_train_encoded, df_test_encoded):
    train_imputed, test_imputed = pp.impute(df_train_encoded, df_test_encoded)

    assert isinstance(test_imputed, np.ndarray) is True, \
        "resulting test object is not a numpy array"
    assert isinstance(train_imputed, np.ndarray) is True, \
        "resulting train object is not a numpy array"


def test_scale_shape(df_train_imputed, df_test_imputed):
    train_array, test_array = pp.scale(df_train_imputed, df_test_imputed)

    assert df_train_imputed.shape == train_array.shape, \
        "Scaling changed the shape of train array"
    assert df_test_imputed.shape == test_array.shape, \
        "Scaling changed the shape of test array"


def test_scale_values(df_train_imputed, df_test_imputed):
    train_array, test_array = pp.scale(df_train_imputed, df_test_imputed)

    assert np.array_equal(train_array, df_train_imputed) is False, \
        "scaler has not transformed the values of train_imputed"
    assert np.array_equal(test_array, df_test_imputed) is False, \
        "scaler has not transformed the values of test_imputed"


def test_scale_dtypes(df_train_imputed, df_test_imputed):
    train_array, test_array = pp.scale(df_train_imputed, df_test_imputed)

    assert isinstance(train_array, np.ndarray) is True, \
        'resulting train object is not a numpy array'
    assert isinstance(test_array, np.ndarray) is True, \
        'resulting test object is not a numpy array'

