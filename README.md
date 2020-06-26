# yeezy_kash

[Yeezy Ka$h](http://54.190.100.173:8501/) is a streamlit web app that help sneaker investors predict whether or not a sneaker will see price surge or price drop for more than 10% in the next 60 days. 

Individual transaction data and product detail were scraped from stockx.com. Feature engineering involves creating rolling metrics of past transaction trends and parsing product details. Features are then feed into lightGBM classifiers. Average testing ROC AUC is 0.84. In the web app users can paste the stockx product page url and select shoe size on a slider, then Yeezy Ka$h will return a graph of historical transaction data of the particular shoe and a price change prediction.


[Google Slides](https://docs.google.com/presentation/d/1DJlCXuTYRNufT9h4LV_Q3JIcVpJSxjAd-OMU92k2HAY/edit?usp=sharing)
[Demo video](https://www.youtube.com/watch?v=18xwPXEI-WU)
[Find me on Linkedin] (https://www.linkedin.com/in/matttso/)
