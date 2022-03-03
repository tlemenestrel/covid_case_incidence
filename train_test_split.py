
import pandas as pd

if __name__ == '__main__':

    start_test = pd.datetime(2020, 11, 1)
    df = pd.read_csv('train_data.csv', index_col='Unnamed: 0').sort_values('date').reset_index(drop=True)
    df['date']=pd.to_datetime(df['date'])
    train_df, test_df = df[df.date<start_test], df[df.date>=start_test]
    train_df.to_csv('temp_train.csv')
    test_df.to_csv( 'temp_val.csv')





