from __future__ import division, print_function
# coding=utf-8
import streamlit as st
import os
import clf_predict
import pandas as pd
import webbrowser
import requests
from bs4 import BeautifulSoup as bs4
from datetime import datetime
import re
import stockx_util as x
from stockx_util import find_between
import plotly.express as px
import plotly.graph_objs as go

###Main program to run streamlit web app to predict sneaker price surge or drop.

#The program scrap last transactions of the queried sneaker/size combination on stockx.com and present
#prediction on the streamlit user interface.

#Usage:
# streamlit run clf_demo.py

def go_to_product_page(item_url):
    """ Fetches product descriptions from product page url.

        Args:
            item_url: url for stocks page

        Returns:
            product_table: dataframe of product properties that includes name, description,
            image url, release date, model, sku and color.

        Raise IndexError:
            when the input product page is invalid
    """
    headers = {
        "referer": "https://stockx.com/",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'"
    }

    req = requests.get(item_url, headers=headers)
    #fname = os.path.join('query', str(datetime.now()) + '.txt')  # used when saving query file to separate flder
    fname = str(datetime.now()) + '.txt'
    if req.status_code == 200:
        soup = bs4(req.content)
        product = soup.findAll('div', 'product-view')
        f = open(fname, "w")
        f.write(str(product))
        f.close()

    f = open(fname, 'r')
    content = f.read()
    f.close()
    # parse content
    try:
        x = content.split('{')[1]
        name = find_between(x, '\"name\":', ',')[1:-1]
        desc = find_between(x, '\"description\":', ',')[1:-1]
        image = find_between(x, '\"image\":', ',')[1:-1]
        release_date = find_between(x, '\"releaseDate\":', ',')[1:-1]
        model = find_between(x, '\"model\":', ',')[1:-1]
        sku = find_between(x, '\"sku\":', ',')[1:-1]
        color = find_between(x, '\"color\":', ',')[1:-1]
        product_table = pd.DataFrame(
            {'name': [name], 'description': [desc], 'image_link': [image], 'release_date': [release_date],
             'model': [model], 'sku': [sku], 'color': [color]})

    except IndexError:
        st.write('Parsing product page failed')

    return product_table


def text_to_df(str):
    """ Helper function for getActivity to parse str into df. Use in getActivity()

           Args:
               str: raw str fetch from get activity

           Returns:
               df that sepeartes each feature into columns

    """

    df = pd.DataFrame()
    record = re.findall('\{.*?\}', str)
    for i in range(len(record)):
        dict = {}
        ss = record[i].split(',')
        for sss in ss:
            ssss = sss.split(':')
            dict[ssss[0][1:-1]] = ssss[1]
        tmp_df = pd.DataFrame(dict, index=[0])
        df = df.append(tmp_df)
    return df

@st.cache
def get_activity(sku, p_name):
    """ Fetches past transactions for a particular product from stockx.com

        Current settings stop fetching at page 3 or no further records (line 135).

           Args:
               sku: sku of product
               p_name: name of product

           Returns:
               activity_df: df of past transaction of the queried product that contains createdAt (time of transaction),
               shoeSize and amount (price).

    """
    base_url = 'http://stockx.com'
    page = 1
    all_activity = ''
    while True:
        item_activity_url = base_url + '/api/products/' + sku + '/activity?state=480&currency=USD&limit=20000&page=' + str(
            page) + '&sort=createdAt&order=DESC&country=US'

        sess = requests.session()
        headers = {
            "referer": base_url + '/' + p_name,
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'"
        }
        req = sess.get(item_activity_url, headers=headers)
        content = req.text
        activity = x.find_between(content, '[', ']')
        all_activity = all_activity + activity
        next_page = re.search('\"nextPage\"\:.*?,', content).group(0).split(':')[1]
        if (next_page == 'null,') or (page == 3): # stop scrapping when page 3 is scrapped or there is no next page
            break
        page += 1

    activity_df = text_to_df(all_activity) # convert scrapped text to df
    # fix data types and extra text
    activity_df['amount'] = activity_df['amount'].astype('float')
    activity_df['createdAt'] = activity_df['createdAt'].str.strip('\"')
    activity_df['shoeSize'] = activity_df['shoeSize'].str.strip('\"')
    activity_df['shoeSize'] = activity_df['shoeSize'].astype('float')
    activity_df['createdAt'] = pd.to_datetime(activity_df['createdAt'])
    # keep only createdAt (time record was created), shoeSize and amount (price)
    activity_df.index = activity_df['createdAt']
    activity_df = activity_df[['shoeSize', 'amount']]
    return activity_df

def plot_shoe(df, p_name, shoe_size):
    """ Plotly express function to plot past transaction history of shoe/size combination

           Args:
               df: df of past transaction
                   must contain: createdAt, amount
                p_name: product name to plot
                shoe_size: shoe size to plot

    """
    df['createdAt'] = df.index
    df = df.sort_index() # make sure time is sorted and in index
    fig = px.line(df, x = 'createdAt', y = 'amount', title = (p_name + '\n Size ' + str(shoe_size)))
    fig.update_layout(
        xaxis_title = 'Transaction Date',
        yaxis_title = 'Transaction Price',
    )
    fig.add_shape(
        # Add horizontal line that indicates that price of the last transaction
        type="line",
        x0= min(df.index),
        y0= df.iloc[-1]['amount'],
        x1= max(df.index),
        y1= df.iloc[-1]['amount'],
        name='Last Transaction Price',
        line=dict(
            color="LightSeaGreen",
            width=4,
            dash="dashdot",
        ),
    )

    fig.add_trace(go.Scatter(
        # add text that serves as the legend for the horizonal line above
        x=[min(df.index)],
        y=[1.05*df.iloc[-1]['amount']],
        text=["              Last\n Transaction"],
        mode="text",
        showlegend= False
    ))

    fig.update_yaxes(tickprefix="$") # $ sign for y label
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black') # bold black line for axis
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig)


def main():
    # main function for the streamlit webapp
    os.environ['TZ'] = 'UTC' # set time zone to UTC

    #####
    # main bodt
    # title
    st.markdown("<h1 style='text-align: left; color: lightseagreen;'>Welcome to Yeezy Ka$h!</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: cornflowerblue;'>Best Kicks for your Buck</h3>", unsafe_allow_html=True)
    st.write("")
    # previously used urls
    #url = 'https://stockx.com/adidas-yeezy-boost-350-v2-white-core-black-red'
    #url = 'https://stockx.com/air-jordan-1-retro-high-bred-toe'
    # url input with default url; go to product page url
    url = ' https://stockx.com/air-yeezy-2-red-october'
    url = st.text_input('Input your StockX URL here', url)
    q_product_info = go_to_product_page(url)

    # slider input of shoe size; TODO: tailor shoe size input to shoe
    q_shoe_size = st.slider("Select shoe size", min_value=4.0, max_value=15.0, value= 13.0, step=0.5)

    # get transaction activity of the shoe
    today = datetime.now()
    activity_df = get_activity(q_product_info['sku'][0], q_product_info['name'][0])
    activity_df.to_csv('test_activity.csv')
    activity_df.index = pd.to_datetime(activity_df.index, utc=True)
    activity_df = activity_df.sort_index()

    # perform prediction
    price_drop, price_rise = clf_predict.predict(q_product_info, q_shoe_size, activity_df)

    # Recommendation
    st.markdown("<h3 style='text-align: left; color: cornflowerblue;'>Recommendation: </h3>", unsafe_allow_html=True)
    if ((price_rise >=0.5) and (price_drop<0.3)):
        st.markdown("According to the model, the price of this shoe will likely <span style='color:green'>**increase by more than 10%**</span> within the next 60 days. It is a <span style='color:green'>**good time to buy this shoe**</span> and there is no rush to sell.", unsafe_allow_html=True)
    elif ((price_rise <0.5) and (price_drop>=0.3)):
        st.markdown("According to the model, the price of this shoe will likely <span style='color:red'>**drop by more than 10%** </span> within the next 60 days. It is <span style='color:red'>**not a good time to buy**</span> this shoe and <span style='color:red'>** there is a rush to sell**</span>.", unsafe_allow_html=True)
    elif ((price_rise <0.5) and (price_drop < 0.3)):
        st.markdown("According to the model, the price of this shoe <span style='color: cornflowerblue;'>**will be neutral **</span>for the next 60 days (+/-) within 10% of current price). Buying is not a bad idea and there is no rush to sell.", unsafe_allow_html=True)
    elif ((price_rise >= 0.5) and (price_drop >= 0.3)):
        st.markdown("According to the model, the price of this shoe <span style='color: deeppink;'>**will be extremely volatile**</span> for the next 60 days with swings going both above and below 10% of the current price. <span style='color:deeppink'>**Buy at your own risk and there is a rush to sell**</span>.", unsafe_allow_html=True)

    # plot shoe transaction data (same size)
    activity_df_shoe_size = activity_df[activity_df['shoeSize']==q_shoe_size]
    plot_shoe(activity_df_shoe_size, q_product_info['name'][0], q_shoe_size)

    # button link to product page
    if st.button('Go to StockX product page'):
            webbrowser.open_new_tab(url)

    ########
    # sidebar
    st.sidebar.title('About the app')
    st.sidebar.markdown(
        """
        Yeezy Ka$h is a web app that can rapidly and accurately predict sneaker price 
        fluctuation on stockx.com.\n
        👉 Paste stockx product page URL on the right to see how Yeezy Ka$h get to work!
        ### Want to learn more about Yeezy Ka$h?
        - Checkout [github](https://github.com/matttso/stockx) repo
        - Checkout my [Linkedin](https://www.linkedin.com/in/matttso/) profile\n
        Created by Matt Tso 2020
        """
    )
    st.sidebar.image(q_product_info['image_link'][0], use_column_width=False) # image of quried product

if __name__ == "__main__":
    main()