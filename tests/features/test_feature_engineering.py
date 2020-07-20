import src.features.feature_engineering as fe
import pandas as pd
import pytest
import datatest as dt


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_train():
    """ loads the example dataset with 1000 rows for testing purpose"""
    df = pd.read_csv("../datafiles_for_tests/testing_set_train.csv")
    return df


def test_add_ratio_features_columns(df_train):
    """ this test checks if the proper ratio colums are added to the dataframe
    after the add_ratio_features was called"""
    df_train = fe.add_ratio_features(df_train)

    assert 'CREDIT_INCOME_PERCENT' in df_train, \
        'CREDIT_INCOME_PERCENT has not been added to dataframe'
    assert 'CREDIT_TERM' in df_train, \
        'CREDIT_TERM has not been added to dataframe'
    assert 'ANNUITY_INCOME_PERCENT' in df_train, \
        'ANNUITY_INCOME_PERCENT has not been added to dataframe'
    assert 'INCOME_BIRTH_PERCENT' in df_train, \
        'INCOME_BIRTH_PERCENT has not been added to dataframe'
    assert 'DAYS_EMPLOYED_PERCENT' in df_train, \
        'DAYS_EMPLOYED_PERCENT has not been added to dataframe'
