import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

def vif_feature_selection(df, threshold):

    data = pd.DataFrame()
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

    return output

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
    