import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.impute import KNNImputer

def impute_df(df, imputer):
    """Imputes each column in the given df
        Usage:
            df: pandas DataFrame to be imputed
            imputer: sklearn-type imputer
    """
    for column in df.columns:
        if df[column].isna().sum() != 0:
            df[column] = imputer.fit_transform(df[[column]])
    return df

def process_data(imputer):
    """Data preprocessing for DengAI. Imputes missing data then generates
    new .csv files for each city 
        Usage:
            imputer: sklearn-type imputer
    """
    df = pd.read_csv('../data/raw/dengue_features_train.csv')
    df_labels = pd.read_csv('../data/raw/dengue_labels_train.csv')
    df_test = pd.read_csv('../data/raw/dengue_features_test.csv')

    df = impute_df(df, imputer)
    df_labels = impute_df(df_labels, imputer)
    df_test = impute_df(df_test, imputer)

    try:
        df.drop(labels=['precipitation_amt_mm', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_min_air_temp_k'], axis=1, inplace=True)
        df_test.drop(labels=['precipitation_amt_mm', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_min_air_temp_k'], axis=1, inplace=True)
    except KeyError:
        print('Columns already dropped. Continuing...')

    sj_train_features = df[df['city'] == 'sj']
    iq_train_features = df[df['city'] == 'iq']

    sj_train_labels = df_labels[df_labels['city'] == 'sj']
    iq_train_labels = df_labels[df_labels['city'] == 'iq']

    sj_test_features = df_test[df_test['city'] == 'sj']
    iq_test_features = df_test[df_test['city'] == 'iq']

    df.to_csv('../data/clean/full/dengue_features_train.csv', index=False)
    df_labels.to_csv('../data/clean/full/dengue_labels_train.csv', index=False)
    df_test.to_csv('../data/clean/full/dengue_features_test.csv', index=False)

    sj_train_features.to_csv('../data/clean/sj/sj_train_features.csv', index=False)
    sj_train_labels.to_csv('../data/clean/sj/sj_train_labels.csv', index=False)

    iq_train_features.to_csv('../data/clean/iq/iq_train_features.csv', index=False)
    iq_train_labels.to_csv('../data/clean/iq/iq_train_labels.csv', index=False)

    sj_test_features.to_csv('../data/clean/sj/sj_test_features.csv', index=False)
    iq_test_features.to_csv('../data/clean/iq/iq_test_features.csv', index=False)

if __name__ == "__main__":
    # Impute using sklearn.impute.KNNImputer
    imputer = KNNImputer()
    process_data(imputer)