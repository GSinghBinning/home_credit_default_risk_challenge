from sklearn.linear_model import LogisticRegression


def logistic_regression(df_train, train_labels, df_test):
    log_regressor = LogisticRegression()
    log_regressor.fit(df_train, train_labels)
    return log_regressor.predict(df_test)