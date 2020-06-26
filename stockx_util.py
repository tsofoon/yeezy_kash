###
# Utility functions for stockx project

import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn import preprocessing
import numpy as np
import streamlit as st

def movecol(df, cols_to_move=[], ref_col='', place='After'):
    # move columns (cols to move) to before or after the reference column
    # df: dataframe that movecol is operating on
    # cols_to_move: columns to move
    # ref_col: reference column
    # place: Before or After. optoins of where cols_to_move is placed relative to ref_col
    # return df after the move_col operation is completed
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



def find_between(s, start, end):
    # return substring between start and end of input string s
    return (s.split(start))[1].split(end)[0]



def save_fig(fig_id, tight_layout=True):
    # utility to save figure
    # fig_id is filename of figure
    path = os.path.join('plots', fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)






# %%
def get_daily_rolling_avg(df, same_size=False):
    # Function to get rolling averages of median daily amount
    # df: input dataframe
    # same_size: flag for operations for same_size (true) or not (any_size)
    # return df with rolling averages computed
    df_daily = df.resample("D").median() # get median daily amount
    days_list = [3, 5, 7, 14, 21, 30, 60] # list of rolling averages days
    # same_size
    if same_size:
        df_daily = df_daily.rename(columns={'amount': 'median_daily_amount_same_size'})
        # compute rolling averages for each day range in days_list
        for d in days_list:
            try:
                df_daily['median_daily_amount_same_size_' + str(d) + 'd_rolling_avg'] \
                    = df_daily['median_daily_amount_same_size'].rolling(d,min_periods = 1).mean()
            except TypeError:
                print('TypeError')
    # any_size
    else:
        df_daily = df_daily.rename(columns={'amount': 'median_daily_amount_any_size'})
        for d in days_list:
            # compute rolling averages for each day range in days_list
            try:
                df_daily['median_daily_amount_any_size_' + str(d) + 'd_rolling_avg'] \
                    = df_daily['median_daily_amount_any_size'].rolling(d).mean()
            except TypeError:
                print('TypeError')
    # only keeping columns that start with median_daily_amount, thus not overlapping with original df
    cols_to_keep = df_daily.columns[df_daily.columns.str.startswith('median_daily_amount')]
    df_daily = df_daily[cols_to_keep]

    # merge_asof with original df
    df = pd.merge_asof(df, df_daily, left_index=True, right_index=True)

    return df


# %%

def get_last_transaction(df):
    # get last transactions of the last shoe and size
    # get df and return the same df after manipulation
    new_df = pd.DataFrame(columns=df.columns)
    for p in tqdm(df['product_name'].unique()): # for each unique shoe and size
        tmp1 = pd.DataFrame(columns=df.columns)
        for s in (df[df['product_name'] == p]['shoe_size'].unique()):
            product_mask, size_mask = (df['product_name'] == p), (df['shoe_size'] == s)
            mask = product_mask & size_mask
            tmp = df[mask]
            tmp['last_amount_same_size'] = tmp['amount'].shift(periods=1) # get last transaction amount (price) same size
            tmp['last_createdAt_same_size'] = tmp['createdAt'].shift(periods=1) # get last created at same size
            tmp['d_last_createdAt_same_size'] = tmp['createdAt'] - tmp['last_createdAt_same_size'] # get time from last created at same size
            try:
                tmp = get_daily_rolling_avg(tmp, same_size=True) # get rolling averages for same size
            except ValueError:
                print('ValueError: ', p, ' ', s)
            tmp1 = tmp1.append(tmp)
        tmp1 = tmp1.sort_index()
        # for each shoe regardless of size
        tmp1['last_amount_any_size'] = tmp1['amount'].shift(periods=1) # get last amount (price) anysize
        tmp1['last_createdAt_any_size'] = tmp1['createdAt'].shift(periods=1) # get last created at any size
        tmp1['last_shoe_size_any_size'] = tmp1['shoe_size'].shift(periods=1) # get last size
        tmp1['d_last_createdAt_any_size'] = tmp1['createdAt'] - tmp1['last_createdAt_any_size'] # get time from last created anysize
        try:
            tmp1 = get_daily_rolling_avg(tmp1, same_size=False)
        except ValueError:
            print('ValueError: ', p)
        new_df = new_df.append(tmp1)
    new_df = new_df.sort_index()
    return new_df

def convert_time_index(df):
    # convert createdAt datetime column of df to int components
    # return same df
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
    # convert datetime and time delta columns of df to int components
    # return same df
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
    # for future (df) get historical transactional data (last transation fo same shoe same size) from old_df
    old_df['createdAt'] = pd.to_datetime(old_df['createdAt'])
    new_df = pd.DataFrame(columns=future.columns)
    # for each shoe and size in future
    for p in tqdm(future['product_name'].unique()):
        tmp1 = pd.DataFrame(columns=future.columns)
        for s in (future[future['product_name'] == p]['shoe_size'].unique()):
            product_mask, size_mask = (future['product_name'] == p), (future['shoe_size'] == s)
            mask = product_mask & size_mask
            tmp = future[mask]
            tmp_old = old_df[(old_df['product_name'] == p) & (old_df['shoe_size'] == s)].tail(1) # get last matching shoe and size from old_df
            col_same_size = tmp_old.columns[tmp_old.columns.str.contains("same_size")].to_list()
            col_same_size.append('product_name')
            tmp = tmp.merge(tmp_old[col_same_size], on='product_name')
            tmp1 = tmp1.append(tmp)
        tmp1 = tmp1.reset_index()
        tmp1 = tmp1.sort_index()
        tmp1_old = old_df[(old_df['product_name'] == p)].tail(1) # get last matching shoe regardless of size from old df
        col_any_size = tmp1_old.columns[tmp1_old.columns.str.contains("any_size")].to_list()
        col_any_size.append('product_name')
        tmp1 = tmp1.merge(tmp1_old[col_any_size], on='product_name')
        new_df = new_df.append(tmp1)
    return new_df

def split_ts (df, t0, t1,t2):
    # split df into old that span from t0 to t1; and future that span from t1 to t2
    old = df[(df.index >= t0) & (df.index<t1)]
    future = df[(df.index >= t1) & (df.index<t2)]
    return old, future

def parse_old_df(old_df):
    # parse old_df to drop duplicated and unused columns. return same
    unused_col = ['name','localAmount','localCurrency', 'chainId','description']
    old_df = old_df.drop(columns = unused_col)
    unused_col = old_df.columns[old_df.columns.str.contains('_y')]
    old_df = old_df.drop(columns = unused_col)
    old_df.columns = old_df.columns.str.rstrip('_x')
    return old_df

def parse_future_df(future_df):
    # parse future_df to drop duplicated and unused columns, return same
    unused_col = ['name','localAmount','localCurrency', 'chainId','description','index']
    future_df = future_df.drop(columns = unused_col)
    unused_col = future_df.columns[future_df.columns.str.contains('_y')]
    future_df = future_df.drop(columns = unused_col)
    future_df.columns = future_df.columns.str.rstrip('_x')
    return future_df

