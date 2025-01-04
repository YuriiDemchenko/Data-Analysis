import pandas as pd

BTC_prices_series = pd.read_csv('crypto_prices.csv')
BTC_prices_series.sort_values(by='BTC-USD Price', ascending=False, inplace=True)
print(BTC_prices_series.sum()/BTC_prices_series.count())
print(BTC_prices_series.mean())