# from sklearn.linear_model import LogisticRegression
#
#
# log_reg = LogisticRegression(max_iter=2000)
#
# log_reg.fit(train_data, train_labels)
#
# log_reg_pred = log_reg.predict_proba(test_data)[:, 1]
#
# submit = df_test.loc[:, ['SK_ID_CURR']]
# submit['TARGET'] = log_reg_pred
#
# submit.to_csv(os.path.join(SUBMISSION_DIRECTORY, 'log_reg1.csv'), index=False)