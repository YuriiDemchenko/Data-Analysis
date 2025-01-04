import pandas as pd

my_series = pd.Series(data = [-10, 100, -30, 50, 100])
my_series = my_series.abs()
my_series = my_series.drop_duplicates()
print(my_series)