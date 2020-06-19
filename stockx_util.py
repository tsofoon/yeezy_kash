import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn import preprocessing
import numpy as np

import streamlit as st

def movecol(df, cols_to_move=[], ref_col='', place='After'):
    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]

    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]

    return (df[seg1 + seg2 + seg3])


# return substring between start and end of input string s
def find_between(s, start, end):
    return (s.split(start))[1].split(end)[0]


# utility to save figure
def save_fig(fig_id, tight_layout=True):
    path = os.path.join('plots', fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)






# %%
def get_daily_rolling_avg(df, same_size=False):
    #df = df.tz_localize('UTC')
    #st.dataframe(df.info())
    #st.dataframe(df.head())
    df_daily = df.resample("D").median()
    days_list = [3, 5, 7, 14, 21, 30, 60]
    if same_size:
        df_daily = df_daily.rename(columns={'amount': 'median_daily_amount_same_size'})
        for d in days_list:
            try:
                df_daily['median_daily_amount_same_size_' + str(d) + 'd_rolling_avg'] \
                    = df_daily['median_daily_amount_same_size'].rolling(d,min_periods = 1).mean()
            except TypeError:
                print('TypeError')
    else:
        df_daily = df_daily.rename(columns={'amount': 'median_daily_amount_any_size'})
        for d in days_list:
            try:
                df_daily['median_daily_amount_any_size_' + str(d) + 'd_rolling_avg'] \
                    = df_daily['median_daily_amount_any_size'].rolling(d).mean()
            except TypeError:
                print('TypeError')
    cols_to_keep = df_daily.columns[df_daily.columns.str.startswith('median_daily_amount')]
    df_daily = df_daily[cols_to_keep]

    df = pd.merge_asof(df, df_daily, left_index=True, right_index=True)

    return df


# %%

def get_last_transaction(df):
    new_df = pd.DataFrame(columns=df.columns)
    for p in tqdm(df['product_name'].unique()):
        tmp1 = pd.DataFrame(columns=df.columns)



        for s in (df[df['product_name'] == p]['shoe_size'].unique()):
            product_mask, size_mask = (df['product_name'] == p), (df['shoe_size'] == s)
            mask = product_mask & size_mask
            tmp = df[mask]
            #print(df.head())
            #print(df.info())
            tmp['last_amount_same_size'] = tmp['amount'].shift(periods=1)
            tmp['last_createdAt_same_size'] = tmp['createdAt'].shift(periods=1)
            tmp['d_last_createdAt_same_size'] = tmp['createdAt'] - tmp['last_createdAt_same_size']
            try:
                tmp = get_daily_rolling_avg(tmp, same_size=True)
            except ValueError:
                print('ValueError: ', p, ' ', s)

            tmp1 = tmp1.append(tmp)

        tmp1 = tmp1.sort_index()

        tmp1['last_amount_any_size'] = tmp1['amount'].shift(periods=1)
        tmp1['last_createdAt_any_size'] = tmp1['createdAt'].shift(periods=1)
        tmp1['last_shoe_size_any_size'] = tmp1['shoe_size'].shift(periods=1)
        tmp1['d_last_createdAt_any_size'] = tmp1['createdAt'] - tmp1['last_createdAt_any_size']
        try:
            tmp1 = get_daily_rolling_avg(tmp1, same_size=False)
        except ValueError:
            print('ValueError: ', p)
        new_df = new_df.append(tmp1)
    new_df = new_df.sort_index()
    return new_df

def convert_time_index(df):
    c = 'createdAt'
    df[c] = df.index
    df[c] = pd.to_datetime(df[c], errors='coerce')
    df[c + '_year'] = df[c].dt.year
    df[c + '_month'] = df[c].dt.month
    df[c + '_day'] = df[c].dt.day
    df[c + '_dow'] = df[c].dt.dayofweek
    df[c + '_weekend'] = (df[c].dt.dayofweek // 4 == 1).astype(float)
    df[c + '_hour'] = df[c].dt.hour
    df[c + '_minute'] = df[c].dt.minute
    del df['createdAt']
    return df

def convert_time(df):
    df['last_createdAt_any_size'] = pd.to_datetime(df['last_createdAt_any_size'])
    df['d_last_createdAt_any_size'] = pd.to_timedelta(df['d_last_createdAt_any_size'])
    print(df['createdAt'].head().dt.month)
    # df = get_dt(df, ['createdAt','release_date','last_createdAt_same_size','last_createdAt_any_size'])
    columns = ['createdAt', 'release_date', 'last_createdAt_same_size', 'last_createdAt_any_size', '']
    for c in columns:

        try:
            df[c] = pd.to_datetime(df[c], errors='coerce')
            df[c + '_year'] = df[c].dt.year
            df[c + '_month'] = df[c].dt.month
            df[c + '_day'] = df[c].dt.day
            df[c + '_dow'] = df[c].dt.dayofweek
            df[c + '_weekend'] = (df[c].dt.dayofweek // 4 == 1).astype(float)
            df[c + '_hour'] = df[c].dt.hour
            df[c + '_minute'] = df[c].dt.minute
        except KeyError:
            print('do nothing')
        except AttributeError:
            print('do nothing')
    df['d_release_date_day'] = df['d_release'].dt.days
    df['d_last_createdAt_day_same_size'] = df['d_last_createdAt_same_size'].dt.days
    df['d_last_createdAt_day_any_size'] = df['d_last_createdAt_any_size'].dt.days
    unused_columns = [ \
        'createdAt', 'release_date', 'd_release', 'd_last_createdAt_same_size', 'last_createdAt_same_size', \
        'last_createdAt_any_size', 'd_last_createdAt_any_size']
    df = df.drop(columns=unused_columns)
    return df

def get_historic_transaction(old_df, future):
    old_df['createdAt'] = pd.to_datetime(old_df['createdAt'])
    new_df = pd.DataFrame(columns=future.columns)
    for p in tqdm(future['product_name'].unique()):
        tmp1 = pd.DataFrame(columns=future.columns)
        for s in (future[future['product_name'] == p]['shoe_size'].unique()):
            product_mask, size_mask = (future['product_name'] == p), (future['shoe_size'] == s)
            mask = product_mask & size_mask
            tmp = future[mask]
            tmp_old = old_df[(old_df['product_name'] == p) & (old_df['shoe_size'] == s)].tail(1)
            col_same_size = tmp_old.columns[tmp_old.columns.str.contains("same_size")].to_list()
            col_same_size.append('product_name')

            tmp = tmp.merge(tmp_old[col_same_size], on='product_name')
            # tmp['d_last_createdAT_same_size'] = tmp['createdAt'] - tmp['last_createdAt_same_size']

            tmp1 = tmp1.append(tmp)
        tmp1 = tmp1.reset_index()
        tmp1 = tmp1.sort_index()
        tmp1_old = old_df[(old_df['product_name'] == p)].tail(1)
        col_any_size = tmp1_old.columns[tmp1_old.columns.str.contains("any_size")].to_list()
        col_any_size.append('product_name')

        tmp1 = tmp1.merge(tmp1_old[col_any_size], on='product_name')
        # tmp1['d_last_createdAT_any_size'] = tmp1['createdAt'] - tmp1['last_createdAt_any_size']
        new_df = new_df.append(tmp1)

    return new_df

def split_ts (df, t0, t1,t2):
    old = df[(df.index >= t0) & (df.index<t1)]
    future = df[(df.index >= t1) & (df.index<t2)]
    return old, future

def parse_old_df(old_df):
    unused_col = ['name','localAmount','localCurrency', 'chainId','description']
    old_df = old_df.drop(columns = unused_col)
    unused_col = old_df.columns[old_df.columns.str.contains('_y')]
    old_df = old_df.drop(columns = unused_col)
    old_df.columns = old_df.columns.str.rstrip('_x')
    return old_df

def parse_future_df(future_df):
    unused_col = ['name','localAmount','localCurrency', 'chainId','description','index']
    future_df = future_df.drop(columns = unused_col)
    unused_col = future_df.columns[future_df.columns.str.contains('_y')]
    future_df = future_df.drop(columns = unused_col)
    future_df.columns = future_df.columns.str.rstrip('_x')
    return future_df


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