import src.model.lgbm_model as lgbm
import pytest
import datatest as dt
import pandas as pd
import numpy as np


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_test():
    """ loads the example test data set, containing 1000 rows and makes it available
    for the testing functions"""
    df = pd.read_csv("../datafiles_for_tests/testing_set_test_model.csv")
    return df


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_train():
    """ loads the example train data set, containing 1000 rows and makes it available
    for the testing functions"""
    df = pd.read_csv("../datafiles_for_tests/testing_set_train_model.csv")
    return df


def test_lightgbm_model_shape_values(df_train, df_test):
    """ this test gets the small example datasets with each 1000 rows,
     to simulate a fast model building process and then to check if the returned
     result matches the expectations"""

    result = lgbm.lightgbm_model(df_train, df_test)
    # the resulting df should contain 1000 rows and 2 columns, one with the keys,
    # and one with the predicted propabilities
    assert result.shape == (1000, 2), \
        "shape of returned df from model differs from expected"
    # Since we are predicting propabilites, the values of the corresponding values
    # should be between 0 and 1
    assert np.any(result.iloc[:, 1].between(0, 1)) == True, \
        " the returned probability column has values, which are not between 0 and 1"
    # Checks if the returned object is a dataframe
    assert isinstance(result, pd.DataFrame) is True, \
        "returned object from model is not a dataframe"
