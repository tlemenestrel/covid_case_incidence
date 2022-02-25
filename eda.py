import pandas as pd

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