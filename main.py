import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

RAW_DIRECTORY = 'data\\raw\\'
PROCESSED_DIRECTORY = 'data\\processed\\'
SUBMISSION_DIRECTORY = 'models\\submissions\\'
import src.data.load as load
import src.data.preprocess as pp
import os
import src.features.feature_engineering as fe

load.load_dataset(RAW_DIRECTORY)

train_data, test_data = load.read_test_train(RAW_DIRECTORY)
df_train = train_data.copy()
df_test = test_data.copy()

train_labels = train_data['TARGET']
train_data = train_data.drop(columns=['TARGET'])


print(train_data.shape, test_data.shape)
train_data = fe.add_ratio_features(train_data)
test_data = fe.add_ratio_features(test_data)





print(train_data.shape, test_data.shape)


train_data, test_data = train_data.align(test_data, join="inner", axis=1)

print(train_data.shape, test_data.shape)


train_data, test_data = pp.encode_categorical(train_data, test_data)
#
train_data, test_data = pp.impute_scale(train_data, test_data)

print(train_data.shape, test_data.shape)

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}


features = [feature for feature in train_data.columns if feature not in ['TARGET','SK_ID_CURR']

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_data[feats], train_data['TARGET'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]



import lightgbm

lgbm_train = lightgbm.Dataset(train_data, train_labels)
lgbm_test = lightgbm.Dataset()
model = lightgbm.train(parameters,
                       train_data,
                       valid_sets = train_labels,
                       num_boost_round = 5000,
                       early_stopping_rounds = 100)


submit = df_test.loc[:, ['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred





#
#
#
#
# # train_data.to_csv(os.path.join(PROCESSED_DIRECTORY, "train_set_processed.csv"), index=False)
# # test_data.to_csv(os.path.join(PROCESSED_DIRECTORY, "test_set_processed.csv"), index=False)
#



