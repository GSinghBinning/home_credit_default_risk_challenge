
def add_ratio_features(df):
    """ Currently only focusing on the application data :
    Applying some common sense, in case of financial classification , for example:
    customers income to credit ratio, or income to birth ratio"""
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['ANNUITY_INCOME_PERCENT'] =  df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['INCOME_BIRTH_PERCENT'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']
    df['DAYS_EMPLOYED_PERCENT'] =  df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    return df


