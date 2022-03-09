import pandas as pd
import numpy as np

def select(treshold, method):
    assert(method in ['shift', 'diff', 'pct_change'])
    df_all = pd.read_csv('train_data.csv', index_col='Unnamed: 0')
    df_all['date'] = pd.to_datetime(df_all['date'])
    feat_labels = df_all.columns[2:-1]


    lag_l = list(range(11)) + [20 + 5 * i for i in range(16)]

    all_lag = pd.DataFrame(index=feat_labels, columns=lag_l)
    res = pd.DataFrame(columns=feat_labels, index=np.unique(df_all.county))

    for lagg in lag_l:
        for county in np.unique(df_all.county):
            df_all_county = df_all[df_all.county == county]
            df_all_county = df_all_county.drop('county', axis=1)
            for f in feat_labels:
                if method == 'shift':
                    cor = df_all_county[f].shift(lagg).corr(df_all_county['response'])
                elif method == 'diff':
                    cor = df_all_county[f].diff(lagg).corr(df_all_county['response'])
                else:
                    cor = df_all_county[f].pct_change(lagg).corr(df_all_county['response'])

                res.loc[county, f] = cor

        all_lag[lagg] = res.abs().mean()

    final_df = all_lag.idxmax(axis=1)[all_lag.max(axis=1) > treshold]

    return final_df


def construct_feat_matrix(method):
    assert (method in ['shift', 'diff', 'pct_change'])
    df_all = pd.read_csv('train_data.csv', index_col='Unnamed: 0')
    df_all['date'] = pd.to_datetime(df_all['date'])
    #feat_labels = df_all.columns[2:-1]

    feat_mat = pd.read_csv('features/{}.csv'.format(method), index_col='Unnamed: 0')
    all_feat = pd.DataFrame(index=df_all.index)

    for county in np.unique(df_all.county):
        df_all_county = df_all[df_all.county == county]
        for f in feat_mat.index:
            lagg = feat_mat.loc[f, '0']
            if method == 'shift':
                tst = df_all_county[f].shift(lagg)
            elif method == 'diff':
                tst = df_all_county[f].diff(lagg)
            else:
                tst = df_all_county[f].pct_change(lagg)


            all_feat.loc[tst.index, f + '_{}_{}'.format(method, lagg)] = tst

    return all_feat

if __name__ == '__main__':

    ### Step 1
    #for method in ['shift', 'diff', 'pct_change']:
    #    df = select(treshold=0.6, method=method)
        #df.to_csv('features/{}.csv'.format(type))
    ## Step 2
    res=[]
    for method in ['shift', 'diff', 'pct_change']:
        df = construct_feat_matrix(method=method)
        res.append(df)
    pd.concat(res, axis=1).to_csv('features/all_feaut.csv')