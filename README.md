# Kaggle Challenge: Home Credit Default Risk

This project aims to solve the Kaggle Challenge from Home Credit Group. To get a good understanding 
about the whole context and detailed informations please have a look on https://www.kaggle.com/c/home-credit-default-risk

## Technical instructions before running the code:

1. Please make sure that all packages are installed (list of packages can be found in _requirements.txt_) <br /> 
   Following command can be used to install the packages:
        
        pip install -r requirements.txt

1. Please provide the data files from Kaggle before running the code. There are **2 options**:  

        Option 1: Copy the unzipped files "application_test.csv" and "application_train.csv" into 
        the following directory: home_credit_default_risk_challenge/data/raw
        
        Option 2: Make sure that the link "DATASET_URL" in the file constants.py does contain a 
        proper working url to the archive.zip file from Kaggle, which contains all the datafiles
   
1. Before running the unittests with pytest please run the command: 
        
        pip install -e .
    To run pytest just  type following command in the terminal (with working directory to the current project):
        
        pytest 


## File Structure of the project

**data** - containing the raw datafiles in _data/raw_ and the preprocessed saved version in _data/processed_

**model** - this is where the results will be saved, after running the code and building the model. 
The _submission_ folder, contains a lgbm_predictions.csv file with the data for the Kaggle submission will be saved after predicting
The model gets saved under _model/lgbm_classifier.txt_ and can be loaded with a command like:

     # Please provide test_features
     if os.path.exists("./model/lgbm_classifier.txt"):
                bst = lgb.Booster(model_file='./model/lgbm_classifier.txt')
                prediction_test = bst.predict(test_features)


**notebooks** - containing the notebook, which was used to do the exploratory analysis

**src** - containing all script files in the correspondent sub folders _data_ (scripts to load, read and preprocess data), 
    _features_ (scripts for feature engineering), _model_ (script with the lgbm model)
    
**tests** - imitating the structure of _src_ and provides unittests with pytest in the same structured files with the prefix:
_test__ as usual in pytest. Additional there is the folder _datafiles\_for\_tests_, which contains small simple datasets,
to perform the unit tests and check the functionality of the various functions. 

## General thoughts about how I addressed this technical task


The main issue was to get a good feeling on how simple I should keep the project, since it was demanded in the task statement,
but on the other hand to provide enough evidence to show my capabilites.

So instead of just using a simple Logistic Regression model, which I implemented at initial stage and which did surprisingly well,
I implemented a Light Gradient Boosting Machine with K-Fold cross validation and focused on showing all main ML concepts 
without going too much in detail. So instead of using all the csv files and do feature engineering on all csv files and
putting all this data into the model building, I focused on the _application..csv_ files and did every 
step, like exploratory analysis, data cleaning and feature engineering in an exemplary manner. 

The focus was to provide a simple understandable project with good structure, unittesting and covering the main concepts of ML models,
like cleaning, feature engineering, encoding, scaling, imputing and modeling. 
Also one part was to show working with git and github, so a dev branch was established and regularly commited to. 
