# import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.metrics import precision_score

def lineplot(df,rotation):
    #create seaborn lineplot
    plot = sns.lineplot(df)
    #rotate x-axis labels
    plot.set_xticklabels(plot.get_xticklabels(), rotation=rotation)
    plt.show()
    
def decomposition(df):
    # Extract and plot trend, seasonal and residuals.
    decomposed = seasonal_decompose(df)
    trend = decomposed.trend
    seasonal = decomposed.seasonal
    residual = decomposed.resid
    plt.figure(figsize=(12,8))
    plt.subplot(411)
    plt.plot(df, label='Original', color='black')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(trend, label='Trend', color='red')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonal', color='blue')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(residual, label='Residual', color='black')
    plt.legend(loc='upper left')
    plt.show()

def Arima_Model(df,col):
    # 1,1,1 ARIMA Model
    arima = ARIMA(df[col], order=(1,1,1))
    ar_model = arima.fit()
    # Forecast
    forecast = ar_model.get_forecast(2)
    ypred = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)
    dp = new_df(ypred,conf_int)
    # Evaluate the model
    print('ARIMA MAE = ', mean_absolute_error(dp.price_actual, dp.price_predicted))

def new_df(ypred,conf_int):
    # creating a new Dataframe dp with the prediction values.
    Date = pd.Series(['2024-01-01', '2024-02-01'])
    price_actual = pd.Series(['184.40','185.04'])
    price_predicted = pd.Series(ypred.values)
    lower_int = pd.Series(conf_int['lower AAPL'].values)
    upper_int = upper_series = pd.Series(conf_int['upper AAPL'].values)
    dp = pd.DataFrame([Date, price_actual, lower_int, price_predicted, upper_int], index =['Date','price_actual', 'lower_int', 'price_predicted', 'upper_int']).T
    dp = dp.set_index('Date')
    dp.index = pd.to_datetime(dp.index)
    return dp

def prep_data():
    data = yf.download("AAPL", start="2000-01-01", end="2022-05-31")
    data['Next_day'] = data['Close'].shift(-1)
    # If the price next day was greater than previous day, we flag it as 1
    data['Target'] = (data['Next_day'] > data['Close']).astype(int)
    return data

def XGBoost_Model(data,position,depth,esti,rs,col):
    # Train test split. Note, this is a time series data.
    train = data.iloc[:position]
    test = data.iloc[position:]
    # Be carefull not to use the next_day feature
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Instantiate a model
    model1 = XGBClassifier(max_depth=depth, n_estimators=esti, random_state=rs)
    # Train the baseline model
    model1.fit(train[features], train[col])
    # Make predictions
    model1_preds = model1.predict(test[features])
    # Convert numpy array to pandas series
    model1_preds = pd.Series(model1_preds, index=test.index)
    # Evaluate the model
    print("the prescision score for the model is: ",precision_score(test['Target'], model1_preds))
    # Plot test['Target'] vs model1_preds
    plt.plot(test[col], label='Actual')
    plt.plot(model1_preds, label='Predicted')
    plt.legend()
    plt.show()
    return model1,features

# First create a predict function for backtesting function
def predict(train, test, features, model):
  model.fit(train[features], train['Target'])
  model_preds = model.predict(test[features])
  model_preds = pd.Series(model_preds, index=test.index, name='predictions')
  combine = pd.concat([test['Target'], model_preds], axis=1)
  return combine

# Create a backtest function
def backtest(data, model, features, start=5031, step=120):
  all_predictions = []
  for i in range(start, data.shape[0], step):
    train = data.iloc[:i].copy()
    test = data.iloc[i:(i+step)].copy()
    model_preds = predict(train, test, features, model)
    all_predictions.append(model_preds)
  return pd.concat(all_predictions)
        
def main():
    # load the dataset.
    # This dataset contains the stock price for 'Apple' and the stock price for 'TXN'
    data = pd.read_csv('AAPL.csv')
    # convert 'Date' to datetime type
    data['Date'] = pd.to_datetime(data['Date'])
    # Univariate analysis - We will only use 'Apple' variable.
    df = data.iloc[:-2,0:2]
    # set the 'Date' column as index
    df = df.set_index('Date')
    lineplot(df['AAPL'],90)
    decomposition(df['AAPL'])
    Arima_Model(df,'AAPL')
    data = prep_data()
    model1,features = XGBoost_Model(data,-30,3,100,42,'Target')
    # backtest
    predictions = backtest(data, model1, features)
    #Evaluate the model
    print("the prescision score for the XGBoost model is: ",precision_score(predictions['Target'], predictions['predictions']))
    

if __name__ =="__main__":
    main()