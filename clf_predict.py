import pandas as pd
import stockx_util as x
import streamlit as st
import lightgbm as lgb
from datetime import datetime, timedelta
import pytz

@ st.cache
def load_data(suffix = '_all_shoe_down10_up10'):
    model_ref = pd.read_csv('model'+suffix+'.csv')
    p_name_ref = pd.read_csv('product_name'+suffix+'.csv')
    c1_ref = pd.read_csv('color1'+suffix+'.csv')
    c2_ref = pd.read_csv('color2'+suffix+'.csv')
    

    return model_ref, p_name_ref, c1_ref, c2_ref
@ st.cache
def get_up_model():
    return lgb.Booster(model_file='down10_up10_best_lgb_model_classify_top_10_shoes_all_features_64_6_features_raise_fold2.txt')
@st.cache
def get_down_model():
    return lgb.Booster(model_file='down10_up10_best_lgb_model_classify_top_10_shoes_all_features_64_all_features_drop_fold2.txt')



def predict(q_product_info, q_shoe_size, activity_df):
    model_ref, p_name_ref, c1_ref, c2_ref = load_data()
    model = str(q_product_info['model'].iloc[0])
    p_name = q_product_info['name'].iloc[0]
    model = model_ref[model_ref['key'] == model]['value'].iloc[0]
    p_name = p_name_ref[p_name_ref['key'] == p_name]['value'].iloc[0]
    release_date = q_product_info['release_date'].iloc[0]
    r = release_date.split('-')
    #st.write(release_date)
    try:
        c1, c2 = q_product_info['color'].iloc[0].split('/')[0],q_product_info['color'].iloc[0].split('/')[1:]
        if len(c2) >1 :
            c2 = '/'.join(c2)
        else: c2 = c2[0]
    except ValueError:
        c1, c2 = q_product_info['color'].iloc[0], 'NaN'

    c1 = c1_ref[c1_ref['key'] == c1]['value'].iloc[0]
    if c2 == 'NaN':
        c2 = 584
    else:
        #st.write(c2)
        c2 = c2_ref[c2_ref['key'] == c2]['value'].iloc[0]


    up_model, down_model = get_up_model, get_down_model



    df_template = pd.DataFrame(columns=['name', 'model', 'color1', 'color2', 'release_date', 'description',
                                        'product_name', 'shoe_size', 'createdAt', 'amount', 'localAmount',
                                        'localCurrency', 'chainId', 'd_release'])

    activity_df = df_template.append(activity_df)
    activity_df['product_name'] = p_name
    activity_df['model'] = model
    activity_df['color1'] = c1
    activity_df['color2'] = c2

    activity_df['createdAt'] = activity_df.index

    activity_df['release_date'] = datetime(int(r[0]), int(r[1]), int(r[2]))

    activity_df['createdAt'] = pd.to_datetime(activity_df.index, errors='coerce')

    activity_df = activity_df.sort_values(by='createdAt')

    activity_df['release_date'] = pd.to_datetime(activity_df['release_date'], utc=True)

    #activity_df['createdAt'] = activity_df['createdAt'].dt.tz_localize(None,nonexistent='shift_backward')
    #activity_df['release_date'] = activity_df['release_date'].dt.tz_localize(None, nonexistent='shift_backward')

    activity_df['d_release'] = activity_df['createdAt'] - activity_df['release_date']
    activity_df.index = activity_df['createdAt']
    activity_df['shoe_size'] = activity_df['shoeSize']
    del activity_df['shoeSize']





    activity_df = x.get_last_transaction(activity_df)
    #activity_df.to_csv('test_activity.csv')
    #st.dataframe(activity_df)
    activity_df = x.parse_old_df(activity_df)
    # print(activity_df.head())

    activity_df = x.convert_time(activity_df)
    activity_df = activity_df.apply(pd.to_numeric, errors='ignore')

    activity_df = activity_df[activity_df['shoe_size'] == q_shoe_size]

    # print(activity_df.head())

    st.dataframe(activity_df.tail())

    features = ['model', 'color1', 'color2', 'product_name', 'shoe_size',
       'last_amount_same_size', 'median_daily_amount_same_size',
       'median_daily_amount_same_size_3d_rolling_avg',
       'median_daily_amount_same_size_5d_rolling_avg',
       'median_daily_amount_same_size_7d_rolling_avg',
       'median_daily_amount_same_size_14d_rolling_avg',
       'median_daily_amount_same_size_21d_rolling_avg',
       'median_daily_amount_same_size_30d_rolling_avg',
       'median_daily_amount_same_size_60d_rolling_avg', 'last_amount_any_size',
       'last_shoe_size_any_size', 'median_daily_amount_any_size',
       'median_daily_amount_any_size_3d_rolling_avg',
       'median_daily_amount_any_size_5d_rolling_avg',
       'median_daily_amount_any_size_7d_rolling_avg',
       'median_daily_amount_any_size_14d_rolling_avg',
       'median_daily_amount_any_size_21d_rolling_avg',
       'median_daily_amount_any_size_30d_rolling_avg',
       'median_daily_amount_any_size_60d_rolling_avg', 'createdAt_year',
       'createdAt_month', 'createdAt_day', 'createdAt_dow',
       'createdAt_weekend', 'createdAt_hour', 'createdAt_minute',
       'release_date_year', 'release_date_month', 'release_date_day',
       'release_date_dow', 'release_date_weekend', 'release_date_hour',
       'release_date_minute', 'last_createdAt_same_size_year',
       'last_createdAt_same_size_month', 'last_createdAt_same_size_day',
       'last_createdAt_same_size_dow', 'last_createdAt_same_size_weekend',
       'last_createdAt_same_size_hour', 'last_createdAt_same_size_minute',
       'last_createdAt_any_size_year', 'last_createdAt_any_size_month',
       'last_createdAt_any_size_day', 'last_createdAt_any_size_dow',
       'last_createdAt_any_size_weekend', 'last_createdAt_any_size_hour',
       'last_createdAt_any_size_minute', 'd_release_date_day',
       'd_last_createdAt_day_same_size', 'd_last_createdAt_day_any_size']#activity_df.columns#.drop(['amount'])#, 'drop20_60d', 'raise20_60d'])
    up_model, down_model = get_up_model(), get_down_model()
    price_drop = down_model.predict(activity_df[features], num_iteration=down_model.best_iteration)
    price_rise = up_model.predict(activity_df[features], num_iteration=up_model.best_iteration)
    price_drop = price_drop[-1]
    price_rise = price_rise[-1]

    return price_drop, price_rise