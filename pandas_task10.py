import pandas as pd

BTC_prices_series = pd.read_csv('crypto_prices.csv')
print(399 in BTC_prices_series.values)
print(399 in BTC_prices_series.round(0).values)