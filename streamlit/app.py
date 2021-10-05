#make the necessary imports 
import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

#define the start date and end date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
#build the wepApp
st.title ("Stock Prediction App")
#AAPL stands for apple stock data
#GOOG stnds for google data
#MSFT stands for microsoft data
#GME stands for gamestop
#stocks to select from

stocks=("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("select dataset for prediction", stocks)

#create a slider
n_years = st.slider("Years of Prediction:", 1,5)
period = n_years *365
#load the data
@st.cache
def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state =st.text("Load Data . . . ")
data = load_data(selected_stock)
data_load_state.text("Loading data. . . done!")

#analyze the data
st.subheader("Raw Data")
#displays the last five rows of the dataset
st.write(data.tail())
def plot_raw_data():
    fig =go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name = "Stock_Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name = "Stock_Close"))
    fig.layout.update(title_text = "Time Series Date", xaxis_rangeslider_visible =True)
    st.plotly_chart(fig)
plot_raw_data()

#forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns = {'Date':"ds", "Close":"y"})
#initialize the model using fbprophet
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)
st.subheader("Forecast Data")
#forecast the last five row data of the dataset
st.write(forecast.tail())

#plot the forecast data
st.write("Forecast Data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)

