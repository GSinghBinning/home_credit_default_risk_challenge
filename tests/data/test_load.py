import src.data.load as ld
import pandas as pd

testing_data_directory = './tests/datafiles_for_tests/'


def test_read_test_train_shape():
    """ This test checks if the read_test_train function works properly and
    if the loaded data objects through the functions match the expected for the
    example data sets (testing_set..)"""
    train_data, test_data = ld.read_test_train(testing_data_directory,
                                               'testing_set_train.csv',
                                               'testing_set_test.csv')
    # train set has one column more, thats why 122
    assert train_data.shape == (1000, 122), \
        "there is some issue with the shape of loaded train data"
    assert test_data.shape == (1000, 121), \
        "there is some issue with the shape of loaded train data"


def test_read_test_train_dtypes():
    """ Checks if the loaded datasets through the read_test_train functions
    are pd.Dataframe's as expected"""
    train_data, test_data = ld.read_test_train(testing_data_directory,
                                               'testing_set_train.csv',
                                               'testing_set_test.csv')
    assert isinstance(train_data, pd.DataFrame) is True, \
        "loaded train data is not a dataframe"
    assert isinstance(test_data, pd.DataFrame) is True, \
        "loaded test data is not a dataframe"


def test_download_dataset_availability():
    """ the function download_dataset needs to check if the demanded datafiles are
    in the according directory, so this test uses the example data sets (testing_set..)
     to test this function and compares it to the associated output if the data is available """
    assert ld.download_dataset(testing_data_directory,
                               "testing_set_train.csv",
                               "testing_set_test.csv") \
           == print("Data is already in directory")
