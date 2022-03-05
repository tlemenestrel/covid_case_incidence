import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge



def modify_feat(df,keep_last_month=False):

    df=pd.get_dummies(df, prefix=['county'], columns=['county'])

    if keep_last_month:
        df['date'] = pd.to_datetime(df['date'])
        start_test = pd.datetime(2020, 10, 29)
        df=df[df.date >= start_test]
    df = df.drop('date', axis=1)
    x,y= df.drop('response', axis=1), df['response']
    #y=np.log(y+1)
    return x,y

def create_model(X_train, y_train, X_test, y_test):
    for alpha in np.linspace(0.2,2, 5):
        #for L1 in  np.linspace(0,1, 5):
            #print("alpha {}, l1 {}".format(alpha,L1 ))
            print("alpha {}".format(alpha))
            #model=make_pipeline(StandardScaler(), ElasticNet(random_state=0, alpha=alpha, l1_ratio=L1))
            #model=make_pipeline( PolynomialFeatures(2), StandardScaler(), ElasticNet(random_state=0, alpha=alpha, l1_ratio=L1))
            model=make_pipeline(StandardScaler(), KernelRidge(alpha=alpha))
            y_pred = model.fit(X_train,y_train ).predict(X_test)
            scr = np.mean(np.abs(np.log(1 + y_pred) - np.log(1 + y_test)))
            #scr = np.mean(np.abs(y_pred, y_test))
            print(scr)

    return scr
def c(X_train, y_train, X_test, y_test):
    alpha = 1
    #model=make_pipeline(StandardScaler(),PolynomialFeatures(2),  ElasticNet(random_state=0, alpha=0.2, l1_ratio=0.5))
    model=make_pipeline(StandardScaler(),  ElasticNet(random_state=0, alpha=1, l1_ratio=0.1))
    y_pred = model.fit(X_train,y_train ).predict(X_test)
    scr = np.mean(np.abs(np.log(1 + y_pred) - np.log(1 + y_test)))
    #scr = np.mean(np.abs(y_pred, y_test))

    return scr




if __name__ == '__main__':


    df_train = pd.read_csv('temp_train.csv', index_col='Unnamed: 0')
    df_test = pd.read_csv('temp_val.csv', index_col='Unnamed: 0')

    X_train, y_train = modify_feat(df_train, keep_last_month=True)
    X_test, y_test = modify_feat(df_test)
    #scr = create_model(X_train, y_train, X_test, y_test)
    scr = c(X_train, y_train, X_test, y_test)
    print(scr)
