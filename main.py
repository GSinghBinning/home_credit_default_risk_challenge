import src.model.lgbm_model as lgbm
import src.data.load as load
import os
import src.features.feature_engineering as fe

# Different directories to save or address the correct files
RAW_DIRECTORY = 'data\\raw\\'
PROCESSED_DIRECTORY = 'data\\processed\\'
SUBMISSION_DIRECTORY = 'models\\submissions\\'

# if the data is not available in the data/raw directory, this function loads it from Kaggle
load.load_dataset(RAW_DIRECTORY)

# read in the csv files into dataframes
train_data, test_data = load.read_test_train(RAW_DIRECTORY)

# adding some ratio_features from the applicant files as a simple example of feature engineering
train_data = fe.add_ratio_features(train_data)
test_data = fe.add_ratio_features(test_data)

# saving the preprocessed data
train_data.to_csv(os.path.join(PROCESSED_DIRECTORY, "train_set_processed.csv"), index=False)
test_data.to_csv(os.path.join(PROCESSED_DIRECTORY, "test_set_processed.csv"), index=False)

# Predicting values through a Light Gradient Boosting model
submission = lgbm.model(train_data, test_data)

# Saving the submission data into a corresponding csv to upload the predictions
submission.to_csv(os.path.join('models\\submissions\\', 'lgbm_model_params_impute_scale.csv'), index=False)


