from datetime import datetime
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

################################################################################
# DATA PRE-PROCESSING
################################################################################

def separate_xy(df, last_col_name):

    X = df.drop([last_col_name], axis=1)
    y = df[last_col_name]
    return X, y

def mean_normalization(df):

    result = df.copy()
    for feature_name in df.columns:
        mean_value = df[feature_name].mean()
        std_value  = df[feature_name].std()
        result[feature_name] = (df[feature_name] - mean_value) / std_value
    return result

def min_max_normalization(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def process_date_and_county(df):

    save_train = df[['date', 'county']]

    # Keep month and day and drop date
    save_train["date"]  = pd.to_datetime(save_train["date"])
    save_train["month"] = save_train['date'].dt.month
    #save_train["day"]   = save_train['date'].dt.month
    save_train = save_train.drop('date', axis=1)

    # Counties
    ### Remove last 3 digits of county
    #save_train['county'] = save_train['county'].astype(str).str[:-3].astype(np.int64)
    save_train = pd.get_dummies(save_train, prefix=['county'], columns=['county'])

    return save_train

def train_test_split(df):

    threshold_date = datetime(2020, 11, 1)
    # Shift the index by 1
    df.index += 1     
    df['date'] = pd.to_datetime(df['date'])

    train_df, val_df = df[df.date<threshold_date], df[df.date>=threshold_date]
    train_df = train_df.drop('Unnamed: 0', axis=1)
    val_df  = val_df.drop('Unnamed: 0', axis=1)

    return train_df, val_df

COL = 'county'
DEFAULT_COLS = [
    'hospital-admissions_smoothed_adj_covid19_from_claims',
]
def add_one_hot_and_interactions(df, interaction_cols=DEFAULT_COLS):
    """
    function to add the one-hot interaction terms
    """
    counties = df['county'].unique().tolist()
    df = pd.get_dummies(df, prefix=[COL], columns=[COL])

    for col in interaction_cols:
        for c in counties:
            colname = f'county_{c}'
            df[f'county_{c}_{col}'] = df[col] * df[colname]

    return df

def add_shifted_features(dataframe, county_list, column_list):
    print('-> Adding shifted features...')
    county_df_list = []
    for county in county_list:

        # Create a separate dataframe for each county
        county_df = dataframe[dataframe['county'] == county]

        for column_name in column_list:
            # Create the new shifted column
            county_df['SHIFT_' + column_name] = county_df[column_name].pct_change(periods=3)
            # Drop the previous one
            county_df = county_df.drop(column_name, axis = 1)

        county_df = county_df.dropna()
        county_df_list.append(county_df)

    print('--> Shifted features added!')

    data = pd.concat(county_df_list)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna()
    data = data.sort_values(by=['date', 'county'])
    return data

################################################################################
# FEATURE SELECTION
################################################################################

def vif_feature_selection(df, threshold):

    output = pd.DataFrame()
    k = df.shape[1]
    vif = [variance_inflation_factor(df.values, j) for j in range(df.shape[1])]

    for i in range(1, k):
        print("Iteration number: ", i)
        print(vif)
        a = np.argmax(vif)
        if (vif[a]<=threshold):
            break
        if (i==1):
            output = df.drop(df.columns[a], axis=1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        elif (i > 1):
            output = output.drop(output.columns[a], axis=1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]

    return list(output)

def get_corr_features(
    df,
    correlation_target,
    correlation_minimum_criteria
    ):

    # Using Pearson Correlation
    cor = df.corr()

    # Correlation with output variable
    target = abs(cor[correlation_target])

    #Selecting and printing highly correlated features
    relevant_features = target[target>correlation_minimum_criteria]
    relevant_features = relevant_features.index.to_list()

    return(relevant_features)

def pearson_corr_mat(
    size_x,
    size_y,
    df,
    correlation_target,
    correlation_minimum_criteria
    ):

    # Using Pearson Correlation

    plt.figure(figsize=(size_x,size_y))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.savefig('pearson_correlation_matrix.png', bbox_inches='tight')
    plt.show()

    # Correlation with output variable

    target = abs(cor[correlation_target])

    #Selecting and printing highly correlated features

    relevant_features = target[target>correlation_minimum_criteria]
    print(relevant_features)

    return(relevant_features)
