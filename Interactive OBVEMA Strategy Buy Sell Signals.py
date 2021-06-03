#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Libraries
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import numpy as np
import plotly.graph_objects as go

plt.style.use('seaborn-pastel')


# In[2]:


stock = input("Enter Ticker: ")


# In[3]:


OBVMovAv1 = int(input("MA: "))


# In[4]:


OBVMovAv2 = int(input("MA: "))


# In[5]:


Start = input(" Year-Month-Day: ")


# In[6]:


df = web.DataReader(stock, data_source='yahoo', start=Start)
df


# In[7]:


OBV = []
OBV.append(0)

for i in range(1, len(df.Close)):
    if df.Close[i] > df.Close[i-1]:
        OBV.append (OBV[-1] + df.Volume[i])
    elif df.Close[i] < df.Close[i-1]:
        OBV.append(OBV[-1] - df.Volume[i])
    else:
        OBV.append (OBV[-1])


# In[8]:


#Get OBV EMA Data
df['OBV'] = OBV
df['OBV_EMA'] = df['OBV'].ewm(span=OBVMovAv1).mean()
df['OBV_EMA1'] = df['OBV'].ewm(span=OBVMovAv2).mean()

OBV_EMA = df['OBV_EMA']
OBV_EMA1 = df['OBV_EMA1']

df = df.dropna()
df


# In[9]:


df['Date'] = pd.date_range(start=Start, periods=len(df), freq='B')


# In[ ]:





# In[10]:


fig = go.Figure([go.Scatter(x=df['Date'], y=df['Close'])])
fig.show()


# In[11]:


from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=df['Date'], y=df['Close'], name="Close"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df['Date'], y=df['OBV_EMA'], name="OBVEMA1"),
    secondary_y=True,
)

fig.add_trace(
    go.Scatter(x=df['Date'], y=df['OBV_EMA1'], name="OBVEMA2"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Double Y Axis Example"
)

# Set x-axis title
fig.update_xaxes(title_text="xaxis title")

# Set y-axes titles
fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)

fig.show()


# In[12]:


#Create a function to signal when to buy and sell an asset
def buy_sell(signal):
  sigPriceBuy = []
  sigPriceSell = []
  flag = -1
  for i in range(0,len(signal)):
    #if MA > MA1  then buy else sell
      if signal['OBV_EMA'][i] > signal['OBV_EMA1'][i]:
        if flag != 1:
          sigPriceBuy.append(signal['OBV_EMA'][i])
          sigPriceSell.append(np.nan)
          flag = 1
        else:
          sigPriceBuy.append(np.nan)
          sigPriceSell.append(np.nan)
        #print('Buy')
      elif signal['OBV_EMA'][i] < signal['OBV_EMA1'][i]:
        if flag != 0:
          sigPriceSell.append(signal['OBV_EMA'][i])
          sigPriceBuy.append(np.nan)
          flag = 0
        else:
          sigPriceBuy.append(np.nan)
          sigPriceSell.append(np.nan)
        #print('sell')
      else: #Handling nan values
        sigPriceBuy.append(np.nan)
        sigPriceSell.append(np.nan)
  
  return (sigPriceBuy, sigPriceSell)


# In[13]:


#Create a new dataframe
signal = pd.DataFrame(index=df['Close'].index)
signal['Close'] = df['Close']
signal['OBV_EMA'] = OBV_EMA
signal['OBV_EMA1'] = OBV_EMA1


# In[14]:


signal


# In[15]:


x = buy_sell(signal)
signal['Buy_Signal_Price'] = x[0]
signal['Sell_Signal_Price'] = x[1]


# In[16]:


signal


# In[17]:


#Daily returns Data
stock_daily_returns = df['Adj Close'].diff()


# In[18]:


df['cum'] = stock_daily_returns.cumsum()
df.tail()


# In[19]:


# MA > MA1 Calculation
df['Shares'] = [1 if df.loc[ei, 'OBV_EMA']>df.loc[ei, 'OBV_EMA1'] else 0 for ei in df.index]


# In[20]:


df['Close1'] = df['Close'].shift(-1)
df['Profit'] = [df.loc[ei, 'Close1'] - df.loc[ei, 'Close'] if df.loc[ei, 'Shares']==1 else 0 for ei in df.index]


# In[21]:


#Profit per Day, and Accumulative Wealth
df['Wealth'] = df['Profit'].cumsum()
df.tail()


# In[22]:


df['diff'] = df['Wealth'] - df['cum']
df.tail()


# In[23]:


df['pctdiff'] = (df['diff'] / df['cum'])*100
df.tail()


# In[24]:


start = df.iloc[0]
df['start'] = start
start['Close']


# In[25]:


start1 = start['Close']
df['start1'] = start1
df


# In[ ]:





# In[26]:


my_stocks = signal
ticker = 'Close'


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])


fig.add_trace(
    go.Scatter(x=df['Date'], y=df['cum'], name="Buy Hold Profit",
    marker=dict(color="Pink"),
    line = dict(color = "#ff425b"),
    stackgroup = "one"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df['Date'], y= df['Wealth'], name='OBVEMA Profit',
    marker=dict(color="Yellow"),
    line = dict(color = "#7eb8fc"),
    stackgroup = "two",
    opacity = 0.6),
    secondary_y=False,
)


fig.add_trace(
    go.Scatter(x=df['Date'], y= my_stocks['Buy_Signal_Price'], name='Buy Signal',
    marker=dict(color="Green", size=12),
    mode="markers"),
    secondary_y=True,
)


fig.add_trace(
    go.Scatter(x=df['Date'], y= my_stocks['Sell_Signal_Price'], name='Sell Signal',
    marker=dict(color="Red", size=12),
    mode="markers"),
    secondary_y=True,
)


fig.add_trace(
    go.Scatter(x=df['Date'], y=df['OBV_EMA1'], name="OBVEMA2",
    marker=dict(color="Blue")),
    secondary_y=True,
)

fig.add_trace(
    go.Scatter(x=df['Date'], y=df['OBV_EMA'], name="OBVEMA1",
    marker=dict(color="Green")),
    secondary_y=True,
)

fig.add_trace(
    go.Scatter(x=df['Date'], y= df['OBV'], name='Real OBV',
    marker=dict(color="Yellow")),
    secondary_y=True,
)


fig.add_trace(
    go.Ohlc(
    x=df['Date'],
    open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'], name='OHLC',
    increasing_line_color= 'cyan', decreasing_line_color= 'gray'),
    secondary_y=False,
)   


# Add figure title
fig.update_layout(
   #title_text="Interactive Buy / Sell"
   title_text=('<b>OBVEMA (1) > OBVEMA (2) Difference Compared to Buy & Hold is : ${:.2f}</b>'.format(df.loc[df.index[-2],'diff']))
)


# Set x-axis title
fig.update_xaxes(title_text="<b>Year</b>")


# Set y-axes titles
fig.update_yaxes(title_text="<b>USD($)</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>On-Balance Volume</b>", secondary_y=True)


fig.update_layout(legend_title_text= 'Stock Close Starts at : ${:.2f}'.format(df.loc[df.index[-2],'start1']))


fig.update_xaxes(rangeslider_visible=True)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

fig.update_xaxes(rangeslider_visible=True)

#fig.update_layout(plot_bgcolor='rgb(203,213,232)')

fig.update_layout( width=1500, height=700)


#fig.update_layout(
#    annotations=[dict(
#        x='2020-03-06', y=0.05, xref='x', yref='paper',
#        showarrow=False, xanchor='left', text='Increase Period Begins')]
#)

fig.show()


# In[ ]:


import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])


app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter


# In[ ]:




