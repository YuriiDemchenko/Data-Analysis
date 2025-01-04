import pandas as pd

BTC_prices_series = pd.read_csv('crypto_prices.csv')
print(BTC_prices_series.squeeze(True))
print(BTC_prices_series.squeeze(False))
print()
print(min(BTC_prices_series.squeeze(True)), max(BTC_prices_series.squeeze(True)))
print()
BTC_prices_series.sort_values(by='BTC-USD Price', inplace=True)
print(BTC_prices_series)
BTC_prices_series.sort_index(inplace=True)
print()
print(BTC_prices_series)