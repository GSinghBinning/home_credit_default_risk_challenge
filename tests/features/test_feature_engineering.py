import src.features.feature_engineering as fe
import pandas as pd
import pytest
import datatest as dt

# @pytest.fixture(scope='module')
# @dt.working_directory(__file__)
# def df_trains():
#     df = pd.read_csv("../datafiles_for_tests/testing_set_train.csv")
#     return df
#
# def test_add_ratio_features_columns():
#     # df = fe.add_ratio_features(pd.dataframe(df_train))
#     print(df_trains)
#     #
#     # assert 'CREDIT_INCOME_PERCENT' in df
#     # assert 'CREDIT_TERM' in df
#     # assert 'ANNUITY_INCOME_PERCENT' in df
#     # assert 'INCOME_BIRTH_PERCENT' in df
#     # assert 'DAYS_EMPLOYED_PERCENT' in df
