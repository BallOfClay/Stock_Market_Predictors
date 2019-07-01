from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quandl
from fbprophet import Prophet
from suppress import suppress_stdout_stderr
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
plt.style.use('fivethirtyeight')
quandl.ApiConfig.api_key = os.environ.get('quandl')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

class Prophet_Model():

    def __init__(self,exchange,symbol,days_to_predict=30):

        self.exchange = exchange.upper()
        self.symbol = symbol.upper()
        self.rows = 1305
        self.get = exchange + '/' + symbol
        self.days_predict = days_to_predict
        self._get_stock()
        
    def plot_hist(self):
        plt.plot(self.stock['Close'])
        plt.title(self.symbol+' Stock History')
        plt.xlabel('Date')
        plt.ylabel('Value (US$)')       

    def _get_stock (self):
        
        if self.rows == -1:
            try:
                self.stock = quandl.get(self.get)
            except Exception as error:
                print('Data was not found, please check exchange and ticker.')
                print(error)
                return
        else:
            try:
                self.stock = quandl.get(self.get,rows=self.rows)
            except Exception as error:
                print('Data was not found, please check exchange and ticker.')
                print(error)
                return
            
        self._fit()
        
    def _fit (self):
        X = self.stock.index
        y = self.stock.Close
        train = pd.DataFrame()
        train['y'] = y.values
        train['ds'] = X.values
        with suppress_stdout_stderr():
            model = Prophet(
                            daily_seasonality=False,
                            weekly_seasonality=False,
                            yearly_seasonality=True,
                            changepoint_prior_scale=.05
                            )
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.fit(train)
            future = model.make_future_dataframe(periods=self.days_predict)
            self.forecast = model.predict(future)
            
    def show(self):
        graph = pd.DataFrame(index=self.stock.index[-self.days_predict:])
        graph['History'] = self.stock[-self.days_predict:].Close
        predicted = self.forecast[-self.days_predict:]
        predicted.set_index('ds',inplace=True)
        fig, ax = plt.subplots()
        plt.autoscale()
        plt.tight_layout(pad=3)
        ax.plot(graph)
        ax.plot(predicted['yhat'])
        ax.legend(['History','Predicted'])
        plt.xticks(rotation=90)
        plt.xlabel('Date')
        plt.ylabel('Value (US$0)')
        plt.title(self.symbol + ' Prophet prediction')
        plt.savefig('static/images/prophet.png')
        
class LSTM_Model():
    
    def __init__(self,exchange,symbol,days_to_predict=30):
        self.exchange = exchange.upper()
        self.symbol = symbol.upper()
        self.rows = 1305
        self.days_predict = days_to_predict
        self.get = exchange + '/' + symbol
        self._get_stock()
        self._scale_and_fit()
        self._predict()
    
    def plot_hist(self):
        plt.plot(self.stock['Close'])
        plt.title(self.symbol+' Stock History')
        plt.xlabel('Date')
        plt.ylabel('Value (US$)')
        
    def _get_stock (self):
        
        if self.rows == -1:
            try:
                self.stock = quandl.get(self.get)
            except Exception as error:
                print('Data was not found, please check exchange and ticker.')
                print(error)
                return
        else:
            try:
                self.stock = quandl.get(self.get,rows=self.rows)
            except Exception as error:
                print('Data was not found, please check exchange and ticker.')
                print(error)
                return
            
    def _scale_and_fit (self):
        data = np.array(self.stock['Close'].values).reshape(-1,1)

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data)
        X_train = [] 
        y_train = []
        for i in range((self.days_predict*2),len(data)-self.days_predict):
            X_train.append(scaled_data[i-(self.days_predict*2):i,0])
            y_train.append(scaled_data[i,0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
    
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, epochs=1, batch_size=32)
        self.model = model
    
    def _predict(self):
        data = self.stock['Close']
        predictions = data[-(self.days_predict*2):]
        for i in range (self.days_predict):
            x = np.array(predictions[-(self.days_predict*2):]).reshape(-1,1)
            scaled_x = self.scaler.fit_transform(x)
            scaled_x = scaled_x.reshape(1,-1,1)
            pred = self.model.predict(scaled_x)
            pred = self.scaler.inverse_transform(pred)
            predictions = predictions.append(pd.Series(pred[0][0]),ignore_index=True)
        df = pd.DataFrame()
        df['Points'] = predictions.values
        dates_index = pd.date_range(self.stock.index[len(self.stock)-self.days_predict],periods=(self.days_predict*3))
        df['dates'] = dates_index
        df = df.set_index('dates')
        self.predictions = df
        
    def show(self):
        hist = self.predictions[:self.days_predict*2]
        pred = self.predictions[-self.days_predict:]
        fig,ax = plt.subplots()
        plt.autoscale()
        plt.tight_layout(pad=3)
        ax.plot(hist)
        ax.plot(pred)
        ax.legend(['History','Predictions'])
        plt.xticks(rotation=90)
        plt.xlabel('Date')
        plt.ylabel('Value (US$0)')
        plt.title(self.symbol + ' LSTM prediction')
        plt.savefig('static/images/lstm.png')
        
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prophet')
def prophet():
    return render_template('prophet.html')

@app.route('/lstm')
def lstm():
    return render_template('lstm.html')

@app.route('/prophet', methods=['POST'])
def prophet_form_post():
    exchange = request.form['exchange']
    stock = request.form['stock']
    days = request.form['days']
    exchange = exchange.upper()
    stock = stock.upper()
    days = days.upper()
    prop = Prophet_Model(exchange,stock,int(days))
    prop.show()
    return render_template('prophet_graph.html')

@app.route('/lstm', methods=['POST'])
def lstm_form_post():
    exchange = request.form['exchange']
    stock = request.form['stock']
    days = request.form['days']
    exchange = exchange.upper()
    stock = stock.upper()
    days = days.upper()
    lstm = LSTM_Model(exchange,stock,int(days))
    lstm.show()
    return render_template('lstm_graph.html')

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
