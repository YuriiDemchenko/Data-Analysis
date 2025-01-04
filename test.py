import pandas as pd

cryptoList = ['BTC', 'ETH', 'XRP', 'XLM', 'ANKR', 'ICP', 'PEPE', 'DOGE', 'SHIB', 'ADA']
cryptoPrices = [45000, 3000, 1.5, 0.5, 0.1, 50, 0.5, 0.3, 0.00001, 2.5]
cryptoSeries = pd.Series(data = cryptoList)
print(cryptoSeries)

cryptoPricesSeries = pd.Series(data = cryptoPrices, index = cryptoList)
print(cryptoPricesSeries)

print('First 3:\n', cryptoPricesSeries.head(3))
print('First 3 most valuable:\n', cryptoPricesSeries.nlargest(3))

stockListDict = {'AAPL': 150, 'GOOGL': 2800, 'AMZN': 3500, 'TSLA': 700}
stockSeries = pd.Series(stockListDict)
print(stockSeries)
print(stockSeries.shape)
print(stockSeries.sum())
print('Last 2 elements in stockSeries:\n', stockSeries.tail(2))
print(stockSeries.memory_usage())

print(150 in stockSeries.values)

crypto_list = ['BTC','XRP','LTC', 'ADA', 'ETH'] 
crypto_series = pd.Series(data = crypto_list)
print(crypto_series.dtype)

crypto_prices = pd.Series(data = [400, 500, 1500, 20, 70])
print(crypto_prices.mean())

my_series = pd.Series(data = [-100, 100, -300, 50, 100])
print(abs(my_series))