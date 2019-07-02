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
I have created a web app that hosts two models for predicting future stock prices the web app can be found at this link ec2-54-201-213-187.us-west-2.compute.amazonaws.com:8080 or in the web_app folder in the code section of this repo. I started this project with the goal of making a prediction model that is able to predict future stock prices based on trends in the stock's history. The web app contains a [Facebook Prophet model](https://facebook.github.io/prophet/) and a neural net LSTM model.
****
### Process

To begin my process I the models that are commonly used to predict stock prices so that I could have an idea of the type of model that I would like to make. During this time I found the FBprophet model and was interested in the results of the model. After running some tests and seeing the results I decided that I wanted to use the model as a benchmark to compare my model's results to. After some more research I saw that the most commonly used model was the LSTM model.

I began writing my code for the model by comparing the previous attempts at stock market prediction before me and using my code to take what worked well from those and attempt to fix what did not work as well. For my data I used the Quandl python API to pull my stock data into each program instead of using a data file with my scripts.
***
### Visualization
***
### Analysis

***
### Conclusion
First, it is very hard to predict the stock market well. The stock market may have some trends and seasonal patteren but there is enough change that it makes models that are trained to predict based on patterns in the data have a very difficult time. My model performed better than I had initially expected.
I would not base any of my financial decisions on predictions made from my (or any model) based on the results that I have seen today. 
The model that I ran my tests on in the src folder is much more accurate than the model that I have hosted on the web app because it uses more data to train and therefore takes longer to train than my web app counterpart. I used less training data on the web app so that it will train faster and not leave the user waiting as long.
***
### Future
In the future I would like to change the way that I run my web in many ways such as:
* use pickled models that are premade and uploaded to a bucket instead of live trainig 
* Make the site look better and improve the usuability of my web app
* Tune my model to get better results
* Implement more models
