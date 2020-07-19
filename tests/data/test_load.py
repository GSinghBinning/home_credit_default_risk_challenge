import src.data.load as ld
import pandas as pd

testing_data_directory = './tests/datafiles_for_tests/'


def test_read_test_train_shape():
    train_data, test_data = ld.read_test_train(testing_data_directory,
                                               'testing_set_train.csv',
                                               'testing_set_test.csv')
    assert train_data.shape == (1000, 122), \
        "there is some issue with the shape of loaded train data"
    assert test_data.shape == (1000, 121), \
        "there is some issue with the shape of loaded train data"


def test_read_test_train_dtypes():
    train_data, test_data = ld.read_test_train(testing_data_directory,
                                               'testing_set_train.csv',
                                               'testing_set_test.csv')
    assert isinstance(train_data, pd.DataFrame) is True, \
        "loaded train data is not a dataframe"
    assert isinstance(test_data, pd.DataFrame) is True, \
        "loaded test data is not a dataframe"


def test_download_dataset_availability():
    assert ld.download_dataset(testing_data_directory,
                               "testing_set_train.csv",
                               "testing_set_test.csv") \
           == print("Data is already in directory")


""" Another test function to check if zip file is downloaded and unzipped """
