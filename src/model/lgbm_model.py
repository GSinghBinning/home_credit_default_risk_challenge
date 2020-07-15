import numpy as np
import pandas as pd
from src.data.preprocess import encode_categorical
from src.data.preprocess import impute_scale
from sklearn.model_selection import KFold
import lightgbm as lgb
import gc


def model(training_features, test_features, n_folds=5):
    """Light gradient boosting model with cross validation.

    Input parameters
        training_features (pd.DataFrame):
            df containing the training_set with Target values.
            test_features (pd.DataFrame):
            df containing the testings features

    Return
        submit (pd.DataFrame):
            df with `SK_ID_CURR` and `TARGET` probabilities of model prediction
    """

    # Extracting ID and Target
    test_id = test_features['SK_ID_CURR']
    training_labels = training_features['TARGET']

    # Deleting the ID and Target columns
    training_features = training_features.drop(columns=['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns=['SK_ID_CURR'])
    training_features, test_features = training_features.align(test_features, join='inner', axis=1)

    # Encoding categorical values and imputing and scaling the dataframes
    training_features, test_features = encode_categorical(training_features, test_features)
    training_features, test_features = impute_scale(training_features, test_features)

    # Create the kfold object
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    # Empty array for test predictions
    prediction_test = np.zeros(test_features.shape[0])

    # For loop for iterating through the defined folds
    for t_index, v_index in k_fold.split(training_features):
        # Training data and validation data for the fold
        train_features, train_labels = training_features[t_index], training_labels[t_index]
        valid_features, valid_labels = training_features[v_index], training_labels[v_index]

        # Building the model
        classifier = lgb.LGBMClassifier(n_estimators=10309, objective='binary',
                                        class_weight='balanced', learning_rate=0.0192,
                                        max_depth=7,
                                        min_child_weight=49,
                                        min_split_gain=0.0803,
                                        num_leaves=33,
                                        reg_alpha=0.1, reg_lambda=0.1,
                                        subsample=0.8, n_jobs=-1, random_state=50)

        # fitting the model
        classifier.fit(train_features, train_labels, eval_metric='auc',
                       eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                       eval_names=['valid', 'train'],
                       early_stopping_rounds=100, verbose=200)

        best_iteration = classifier.best_iteration_

        # Make predictions
        prediction_test += classifier.predict_proba(test_features, num_iteration=best_iteration)[:, 1] / k_fold.n_splits

        # Clean up memory
        gc.enable()
        del classifier, train_features, valid_features
        gc.collect()

    # creat the result dataframe for the submission
    submit = pd.DataFrame({'SK_ID_CURR': test_id, 'TARGET': prediction_test})


    return submit
