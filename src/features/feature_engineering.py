


""" Applying some common sense, in case of financial classification , for example:
customers income to credit ratio, or income to birth ratio"""


def add_ratio_features(df):

    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
   # df['CREDIT_TO_GOODS_PERCENT'] =  round(df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'],10)

    df['ANNUITY_INCOME_PERCENT'] =  df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    #df['INCOME_EMPLOYED_PERCENT'] =  round(df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED'],10)
    df['INCOME_BIRTH_PERCENT'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']
    df['DAYS_EMPLOYED_PERCENT'] =  df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    return df


def add_poly_features(df,features):
    poly_transformer = PolynomialFeatures(degree = 3)
    poly_transformer.fit_transform()



