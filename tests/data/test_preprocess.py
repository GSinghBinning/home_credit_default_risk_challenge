import src.data.preprocess as pp
import pandas as pd
import pytest
import datatest as dt
import numpy as np


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_test():
    """ loads a simple dataset with 1000 rows imitating the application_test.csv"""
    df = pd.read_csv("../datafiles_for_tests/testing_set_test.csv")
    return df


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_train():
    """ loads a simple dataset with 1000 rows imitating the application_train.csv"""
    df = pd.read_csv("../datafiles_for_tests/testing_set_train.csv")
    return df


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_train_encoded():
    """ loads a encoded version of the simple train dataset, to perform imputing """
    df = pd.read_csv('../datafiles_for_tests/testing_set_train_encoded.csv')
    return df


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_test_encoded():
    """ loads a encoded version of the simple test dataset, to perform imputing """
    df = pd.read_csv('../datafiles_for_tests/testing_set_test_encoded.csv')
    return df


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_test_imputed():
    """ loads a imputed version of the simple encoded test datasetas an array, to perform scaling"""
    test_array = np.loadtxt('../datafiles_for_tests/testing_set_test_imputed.csv', delimiter=',')
    return test_array


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_train_imputed():
    """ loads a imputed version of the simple encoded train dataset as an array, to perform scaling"""
    train_array = np.loadtxt('../datafiles_for_tests/testing_set_train_imputed.csv', delimiter=',')
    return train_array


def test_encode_categorical_dtypes(df_train, df_test):
    """ this function tests the data types of the columns of the returned dataframes from the
     encoding function, there shouldn't be any columns with the datatype object anymore """
    df_train, df_test = df_train.align(df_test, join='inner', axis=1)
    df_train_result, df_test_result = pp.encode_categorical(df_train, df_test)

    for column in df_train_result:
        # Check columns of returned train dataframe
        assert df_train_result[column].dtype != "object", \
            "Training set has still object column after categorical encoder"
        # Check columns of returned test dataframe
        assert df_test_result[column].dtype != "object", \
            "Test set has still object column after categorical encoder"


def test_encode_categorical_dtypes_shape(df_train, df_test):
    """ This function makes sure that the returned df from the encoder, have the correct shape"""

    # Since the train set has the TARGET column, both df has to be aligned
    df_train, df_test = df_train.align(df_test, join='inner', axis=1)
    # Get the result dataframes from the encoder
    df_train_result, df_test_result = pp.encode_categorical(df_train, df_test)
    # Make sure that the shape of the resulting df do not differ from the original df shape
    assert df_train_result.shape == df_test_result.shape, \
        "Shape of the dataframes is not aligned"


def test_data_cleaning_application_values(df_train):
    """Makes sure that the cleaning function does it job and deletes faulty values
    in CODE_GENDER and outliers in DAYS_EMPLOYED"""

    df_train = pp.data_cleaning_application(df_train)
    # Check if Code Gender has just two unique values
    assert df_train.CODE_GENDER.nunique() == 2, \
        "The amount of unique CODE_GENDER doesn't match the number 2 "
    # Check if there are still some 365243 values in DAYS_EMPLOYED
    assert (df_train['DAYS_EMPLOYED'] == 365243).sum() == 0, \
        "There are still some faulty values(365243) in DAYS_EMPLOYED"


def test_data_cleaning_application_shape(df_train):
    """ Checks the shape of the returned dataframe after datacleaning,
    since the cleaning deletes one row in the example data,
    the shape should match (999,122)"""

    df_train = pp.data_cleaning_application(df_train)
    assert df_train.shape == (999, 122), \
        "The resulting shape of df doesn't match the expected"


def test_impute_shape(df_train_encoded, df_test_encoded):
    """ checks the shape of the returned data object after imputing the nan values,
    it should not differ from the originall data object"""
    df_train_imputed, df_test_imputed = pp.impute(df_train_encoded, df_test_encoded)

    # testing shape for the train set and train set
    assert df_train_encoded.shape == df_train_imputed.shape, \
        "Imputing changed the shape of train dataframe"
    assert df_test_encoded.shape == df_test_imputed.shape, \
        "Imputing changed the shape of test dataframe"


def test_impute_values(df_train_encoded, df_test_encoded):
    """ checks if after using the imputer there are still some NAN values in the data """
    df_train_imputed, df_test_imputed = pp.impute(df_train_encoded, df_test_encoded)

    # there should not be any NAN values anymore
    assert np.isnan(np.sum(df_train_imputed)) == False, \
        "there are still some NAN values in the train dataframe"
    assert np.isnan(np.sum(df_test_imputed)) == False, \
        "there are still some NAN values in the test dataframe"


def test_impute_dtypes(df_train_encoded, df_test_encoded):
    """ the resulting data object after imputing should be a an array,
    since the other functions, which follow afterwards expect an array, so this test
    checks if its an array or not """
    train_imputed, test_imputed = pp.impute(df_train_encoded, df_test_encoded)

    assert isinstance(test_imputed, np.ndarray) is True, \
        "resulting test object is not a numpy array"
    assert isinstance(train_imputed, np.ndarray) is True, \
        "resulting train object is not a numpy array"


def test_scale_shape(df_train_imputed, df_test_imputed):
    """ scaling should not change the shape of the data objects, so this test
    is used to check the shape before and after the scaling"""
    train_array, test_array = pp.scale(df_train_imputed, df_test_imputed)

    assert df_train_imputed.shape == train_array.shape, \
        "Scaling changed the shape of train array"
    assert df_test_imputed.shape == test_array.shape, \
        "Scaling changed the shape of test array"


def test_scale_values(df_train_imputed, df_test_imputed):
    """Scaling should return different values,so this test checks if the original
    data object and returned data object are equal or not"""
    train_array, test_array = pp.scale(df_train_imputed, df_test_imputed)

    # it should not be equal, thats why "is False"
    assert np.array_equal(train_array, df_train_imputed) is False, \
        "scaler has not transformed the values of train_imputed"
    assert np.array_equal(test_array, df_test_imputed) is False, \
        "scaler has not transformed the values of test_imputed"


def test_scale_dtypes(df_train_imputed, df_test_imputed):
    """scaling should not change the data type of returned object, so this test checks if
    the returned object is still a an array"""
    train_array, test_array = pp.scale(df_train_imputed, df_test_imputed)

    assert isinstance(train_array, np.ndarray) is True, \
        'resulting train object is not a numpy array'
    assert isinstance(test_array, np.ndarray) is True, \
        'resulting test object is not a numpy array'

