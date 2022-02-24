def build_pearson_corr_mat(
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