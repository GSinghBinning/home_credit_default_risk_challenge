import dload
import os
import pandas as pd
import constants as c


def download_dataset(directory_path, train_set='application_train.csv', test_set='application_test.csv'):
    """ This function downloads the kaggle dataset, in case if it doesn't exist.
    In case the dataset is not available, it downloads the dataset from the provided url,
    unzipps the downloaded archive.zip file and saves the files to the directory data/raw
    and then deletes the downloaded zip file. """

    # checks if the two files provided through train_set and test_set are available
    if not (os.path.exists(os.path.join(directory_path, test_set)) and
            os.path.exists(os.path.join(directory_path, train_set))):
        print("Downloading the datasets from Kaggle ...")
        dload.save_unzip(c.DATASET_URL, directory_path, delete_after=True)
    else:
        # if the data exists just prints this message
        print("Data is already in directory")


def read_test_train(directory_path, train_set, test_set):
    """ Function used to read in the application test and train csv files
    and save the data as pd.Dataframe. """
    train_data = pd.read_csv(os.path.join(directory_path, train_set))
    test_data = pd.read_csv(os.path.join(directory_path, test_set))

    return train_data, test_data
