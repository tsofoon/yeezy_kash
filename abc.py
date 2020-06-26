#unused scribble

pred.insert(0, 'time', 'NaN')
for d in [0]:
    input_date = datetime.utcnow() + relativedelta(days=d)
    pred_tmp = pd.DataFrame(columns=pred.columns)
    data_dict = {}
    for s in pred_tmp.columns:
        if s == 'time':
            data_dict[s] = [input_date]
        else:
            amount_same_size = activity_df[(activity_df['shoeSize'] == s)]
            try:
                amount_same_size_last_day = amount_same_size[amount_same_size.index == max(amount_same_size)]
                median_daily_amount_same_size = amount_same_size_last_day['amount'].median()
                last_amount_same_size = amount_same_size_last_day['amount'].head(1)[0]
                last_amount_same_size = amount_same_size_last_day['amount'].head(1)[0]
                last_amount_any_size = activity_df['amount'].head(1)[0]
                last_createdAt_same_size = amount_same_size_last_day['amount'].index[0]
                curr_pred_price = \
                predict.streamlit_predict(input_date, q_product_info, s, median_daily_amount_same_size, \
                                          last_amount_same_size, last_createdAt_same_size, last_amount_any_size)[0]
            except IndexError:
                curr_pred_price = predict.streamlit_predict(input_date, q_product_info, s, np.nan,
                                                            np.nan, np.nan)[0]
            data_dict[s] = curr_pred_price  # "{:.2f}".format(curr_pred_price)
    pred_tmp = pd.DataFrame(data_dict)
    pred = pred.append(pred_tmp)
pred = pred.reset_index().drop(columns=['index'])

# st.dataframe(activity_df.head())
# st.dataframe(activity_df[activity_df['shoeSize'] == q_shoe_size].head())
activity_df = activity_df[(activity_df['shoeSize'] >= min(sizes)) & (activity_df['shoeSize'] <= max(sizes))]
# st.dataframe(activity_df)
plot_shoe(activity_df, q_product_info['name'][0], q_shoe_size)
st.pyplot()
plt.plot(pred.iloc[:, 0], pred.iloc[:, 1:4], linestyle='dashed', linewidth=0.5)
plt.plot(pred.iloc[:, 0], pred.iloc[:, 4], linewidth=3)
plt.plot(pred.iloc[:, 0], pred.iloc[:, 5:], linestyle='dashed', linewidth=0.5)
plt.legend(pred.iloc[:, 1:].columns)
plt.title('Predicted Price for ' + str(q_product_info['name'][0]))
plt.ylabel('Predicted Price (USD)')
plt.xlabel('Days from now')
plt.xticks(rotation=45)

model_ref, p_name_ref, c1_ref, c2_ref, old_data = load_data()
model = str(q_product_info['model'].iloc[0])
p_name = q_product_info['name'].iloc[0]
model = model_ref[model_ref['key'] == model].iloc[0]

p_name = p_name_ref[p_name_ref['key'] == p_name].iloc[0]
# st.write(q_product_info['color'][0].split('/'))
try:
    c1, c2 = q_product_info['color'].iloc[0].split('/')
except ValueError:
    c1, c2 = q_product_info['color'].iloc[0], 'NaN'
st.write(p_name)
st.write(model, p_name, c1, c2)
st.dataframe(c1_ref)
st.write(c1)
c1 = c1_ref[c1_ref['key'] == c1]['value'].iloc[0]
if c2 == 'NaN':
    c2 = 584
else:

    c2 = c2_ref[c2_ref['key'] == c2]['value'].iloc[0]
st.write(model, p_name, c1, c2)

up_model, down_model = get_up_model, get_down_model

old_data = old_data[old_data['product_name'] == p_name]
st.dataframe(old_data)
st.dataframe(curr_data)

old_data = old_data[(old_data['product_name'] == 8)&(old_data['shoe_size'] == shoe_size)].head(1)
    old_data['median_daily_amount_same_size'] = curr_data.iloc[-1]['amount']
    old_data['median_daily_amount_any_size'] = curr_data.iloc[-1]['amount']
    old_data['last_amount_same_size'] = curr_data.iloc[-1]['amount']
    old_data['last_amount_any_size'] = curr_data.iloc[-1]['amount']
    q_product_info.to_csv('q_product_info.csv')