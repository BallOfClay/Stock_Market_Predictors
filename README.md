# Stock Market Predictors
## Capstone 2

![alt text](https://github.com/shillis17/Stock_Market_Predictors/blob/master/img/head.png)
 #### Seth Hillis | Galvanize DSI | July 2 2019
 
 Table of Contents
<!--ts-->
 * [Description](#description)
 * [Process](#process)
 * [Visualization](#visualization)
 * [Analysis](#analysis)
 * [Conclusion](#conclusion)
 * [Future](#future)
<!--te-->
****

### Description
I have created a web app that hosts two models for predicting future stock prices. The web app can be found at this link: ec2-54-201-213-187.us-west-2.compute.amazonaws.com:8080 or in the web_app folder in the code section of this repo. I started this project with the goal of making a prediction model that is able to predict future stock prices based on trends in the stock's history. The web app contains a [Facebook Prophet model](https://facebook.github.io/prophet/) and a neural net LSTM model.
****
### Process

To begin my process I researched the models that are commonly used to predict stocks so I could limit which models I was considering. During this time I found the FBprophet model and was interested in how the model works and how well it performs. After running some tests and seeing the results, I decided to use the model as a benchmark to compare my model's results to. After some more research, I saw that the most commonly used model was the LSTM model.

I began writing my code for the model by comparing the previous attempts at stock market prediction and using my code to take what worked well from those and attempt to fix what did not work as well. For my data I used the Quandl python API to pull my stock data into each program instead of using a data file with my scripts.
***
### Visualization
Here are some of the stocks that I have tested so that you can see the results of each model.
#### MSFT
![alt text](https://github.com/shillis17/Stock_Market_Predictors/blob/master/img/msft.png)
#### AAPL
![alt text](https://github.com/shillis17/Stock_Market_Predictors/blob/master/img/aapl.png)
#### JNJ
![alt text](https://github.com/shillis17/Stock_Market_Predictors/blob/master/img/jnj.png)
### Analysis
|Stock|LSTM|Prophet|
|-----|----|-------|
|MSFT|16.74|9.91|
|AAPL|37.33|54.90|
|JNJ|17.29|6.35|

The prophet model seems to have a better rmse for most of the stock I have tested. However, it appears that my lstm is less prone to strange occurances, such as in AAPL when the stock was split. The prophet model predicted a harsh drop, however my lstm model did not and was actually better for that specific stock.
***
### Conclusion
First, it is very hard to predict the stock market well. The stock market may have some trends and seasonal patterns but there is enough change that it makes models that are trained to predict based on patterns in the data have a very difficult time. My model performed better than I had initially expected.
I would not base any of my financial decisions on predictions made from my model (or any model) based on the results that I have seen today. 
The model that I ran my tests on in the src folder is much more accurate than the model that I have hosted on the web app, because it uses more data to train and therefore takes longer to train than my web app counterpart. I used less training data on the web app so that it will train faster and not leave the user waiting as long.
My final conclusion for my project is no, I cannot predict the stock market because it is unpredictable and has many more factors than just historical trends and patterns.
***
### Future
In the future I would like to change the way that I run my web in many ways such as:
* use pickled models that are premade and uploaded to a bucket instead of live training 
* Make the site look better and improve the usuability of my web app
* Tune my model to get better results
* Implement more models
