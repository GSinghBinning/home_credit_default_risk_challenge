import dload
import os
import pandas as pd
import gc

""" This function downloads the kaggle dataset, in case if it doesn't exist.
If the shorturl is broken for whatever reason, the full link is commented out 
underneath the function and should be used.
"""


def load_dataset(directory_path):
    if not (os.path.exists(os.path.join(directory_path, 'application_test.csv')) and
            os.path.exists(os.path.join(directory_path, 'application_train.csv'))):
        url = "shorturl.at/bkrJN"
        dload.save_unzip(url, directory_path)
        os.remove("archive.zip")
    else:
        print("Data is already in directory")


# https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/9120/860599/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1594819922&Signature=HERl109nv67QCae9gXcex2zL1hs4iWWkjHUZUGXngverIdShIWVPUzdThhmDibjQpZ7WAUCj3MhLjKKZG%2FTp5y3f8Ne2DnSOA97jHE7RmZ%2FbA0dLsJRkEP3dQErGNCtoUbP85uwejNauxhcjTyPIj%2FeuBpZ0WJCIM%2BYN4Wggy0cwrNi2jLRHIomV79wCtAImCe0LMJ%2BeEY4PnnBXRrDcVVee0Ldl9PMboLmDyX5yA1c0nTigsg%2BVS2mQvc7d7zMKoA3kZGXItJHYfylZgH4mdvn9Z%2BFYN7hWJdGx1gjuIOyEwQBcHJN8bC2eXYFUSHb2FLPa0i9a6Vuijk6jLcQSEQ%3D%3D&response-content-disposition=attachment%3B+filename%3Dhome-credit-default-risk.zip

""" Function used to read in the application test and train data. """


def read_test_train(directory_path):
    train_data = pd.read_csv(os.path.join(directory_path, 'application_train.csv'))
    test_data = pd.read_csv(os.path.join(directory_path, 'application_test.csv'))
    df = train_data.append(test_data)
    del train_data, test_data
    gc.collect()
    return df
