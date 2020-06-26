import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

def get_rolling_result(df):
    out_df = pd.DataFrame(columns=df.columns)
    for p in df['product_name'].unique():
        tmp1_df = pd.DataFrame(columns=df.columns)
        for s in df[df['product_name'] == p]['shoe_size'].unique():
            tmp_df = df[(df['product_name'] == p) & (df['shoe_size'] == s)]
            tmp_df['amount'] = tmp_df['amount'].rolling(2).mean()
            tmp1_df = tmp1_df.append(tmp_df)
        out_df = out_df.append(tmp1_df)

    return out_df


def get_min_result(df):
    out_df = pd.DataFrame(columns=df.columns)
    for p in df['product_name'].unique():
        tmp1_df = pd.DataFrame(columns=df.columns)
        for s in df[df['product_name'] == p]['shoe_size'].unique():
            tmp_df = df[(df['product_name'] == p) & (df['shoe_size'] == s)]
            tmp_df['amount'] = tmp_df['amount'].min()
            tmp1_df = tmp1_df.append(tmp_df)
        out_df = out_df.append(tmp1_df)

    return out_df.apply(pd.to_numeric, errors='ignore')

def MAPE(y_true, y_pred):
    #y_true, y_pred = check_arrays(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true))

def get_metrics(y_pred,y_true):
    print(mean_squared_error(y_pred,y_true,squared = False),MAPE(y_pred, y_true))
    return mean_squared_error(y_pred,y_true,squared = False),MAPE(y_pred, y_true)

def plot_scatter_error(df, x,y,s,c):
    plt.scatter(df[x],df[y], s , c = df[c])
    plt.colorbar()
    plt.ylabel(y)
    plt.xlabel(x)
    #plt.xlim(200,900)

def plot_line_error(df, x,y,s):
    for c in df[s].unique():
        try:
            plt.plot(df[df[s]==c][x],df[df[s]==c][y], label = c)
        except KeyError:
            print('Key Error:' ,c)
    #plt.xlim(200,900)