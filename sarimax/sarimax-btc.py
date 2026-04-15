from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sys
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.api import VAR
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import logging
import sys
import os
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from fitter import Fitter, get_common_distributions
import ruptures as rpt
import statsmodels.tsa.stattools as ts
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from IPython.display import display
except ImportError:
    display = print

from template.prelude_template import load_data
from template.prelude_template import load_polymarket_data

df_btc = load_data()
print(df_btc.shape)

polymarket_data = load_polymarket_data()
print('keys',polymarket_data.keys())

df_markets = polymarket_data["markets"]
df_markets['diff'] = (df_markets['end_date'] - df_markets['created_at']).dt.days
df_markets=df_markets[df_markets['diff'] >= 7]
df_markets=df_markets[df_markets['volume'] > 100000]


df_politics_trades=polymarket_data["trades"]
df_politics_tokens=polymarket_data["tokens"]

trade1=pd.merge(df_politics_trades, df_markets, on='market_id', how='inner')
trade1=pd.merge(trade1, df_politics_tokens, on=['market_id', 'token_id'], how='inner')
trade1['trade_date'] = trade1['timestamp'].dt.strftime('%Y-%m-%d')
trade1['trade_date'] = pd.to_datetime(trade1['trade_date'])
trade1['expected_size'] = trade1['size']*trade1['price']
allowed_outcomes = ['Up','Yes','Positive','up','yes','positive']
trade1=trade1[trade1['outcome'].isin(allowed_outcomes)]


#We get data for year=2025
year=2025


trades_yyyy = trade1[trade1['timestamp'].dt.year == year]
sell_trades_yyyy = trades_yyyy[trades_yyyy['side'] == 'SELL']
buy_trades_yyyy = trades_yyyy[trades_yyyy['side'] == 'BUY']

buy_trades_summary_yyyy = buy_trades_yyyy.groupby(['market_id','trade_date']).agg({
    'expected_size': ['sum'],
    'price': 'mean',
    'size': 'sum'
})

sell_trades_summary_yyyy = sell_trades_yyyy.groupby(['market_id','trade_date']).agg({
    'expected_size': ['sum'],
    'price': 'mean',
    'size': 'sum'
})

trades_summary_yyyy=pd.merge(buy_trades_summary_yyyy, sell_trades_summary_yyyy, left_index=True, right_index=True)
trades_summary_yyyy.rename(columns={"expected_size_x":"expected_buy_size","price_x":"buy_price","size_x":"buy_size",
                                                          "expected_size_y":"expected_sell_size","price_y":"sell_price","size_y":"sell_size"},inplace=True)



trades_summary_yyyy_flat = pd.DataFrame({
    'expected_buy_size': trades_summary_yyyy['expected_buy_size']['sum'],

    'buy_price': trades_summary_yyyy['buy_price']['mean'],
    'buy_size': trades_summary_yyyy['buy_size']['sum'],
    'expected_sell_size': trades_summary_yyyy['expected_sell_size']['sum'],
    'sell_price': trades_summary_yyyy['sell_price']['mean'],
    'sell_size': trades_summary_yyyy['sell_size']['sum'],
    
})

trades_summary_yyyy_flat.rename(columns={"expected_size_x":"expected_buy_size","price_x":"buy_price","size_x":"buy_size",
                                                          "expected_size_y":"expected_sell_size","price_y":"sell_price","size_y":"sell_size"},inplace=True)

trades_summary_yyyy_flat.reset_index(inplace=True)

trades_summary_yyyy_flat2=pd.merge(trades_summary_yyyy_flat, df_markets, on='market_id', how='left')
trades_summary_yyyy_flat2['end_date'] = trades_summary_yyyy_flat2['end_date'].dt.date
trades_summary_yyyy_flat2['end_date'] = pd.to_datetime(trades_summary_yyyy_flat2['end_date'], format='%Y-%m-%d')


START_DATE="2025-12-01"
END_DATE="2025-12-31"

x=trades_summary_yyyy_flat2[['market_id','trade_date','buy_price','question']]
x=x[(x['trade_date'] >= START_DATE) & (x['trade_date'] <= END_DATE)] 
#print(x.head(2))

pivot = pd.pivot_table(x, 
                       values='buy_price', 
                       index='trade_date', 
                       columns='market_id', 
                       aggfunc='sum',
                       fill_value=0) # Replaces NaNs with 0
display(pivot.head(2))



#Get btc data for 2025
df_btc_2025=df_btc[df_btc.index.year == 2025]
display(df_btc_2025.head(2))


#Merge btc data with pivot
btc_small = df_btc_2025['PriceUSD'].to_frame()
btc_small = btc_small.loc[START_DATE:END_DATE]
df=pd.merge(btc_small, pivot, left_index=True, right_index=True)

display("df...")
display(df.head(2))

#Exclude the PriceUSD columns
exog_matrix = df.loc[:, ~df.columns.isin(['PriceUSD'])]
# Fit model (Order 1,1,1 handles the price trend and momentum)
model = SARIMAX(df["PriceUSD"], exog=exog_matrix, order=(1, 1, 1))
results = model.fit(disp=False)
sorted_series = results.params.sort_values(ascending=False)



values=[]
heads = sorted_series.head(50).index.values
values.append(heads.flatten().tolist())
values = np.array(values).flatten()
filtered_df = df_markets[df_markets['market_id'].isin(values)]
display('Top 10 markets with positive correlation period','Start=',START_DATE,'End=',END_DATE) 
display(filtered_df['question'].head(10))


tails = sorted_series.tail(50).index.values
#heads = sorted_series.head(50).index.values
values=[]
#values.append(heads.flatten().tolist())
values.append(tails.flatten().tolist())
values = np.array(values).flatten()
filtered_df = df_markets[df_markets['market_id'].isin(values)]
display('Top 10 markets with negative correlation','Start=',START_DATE,'End=',END_DATE) 


display(filtered_df['question'].head(10))
