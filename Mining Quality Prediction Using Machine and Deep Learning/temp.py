import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile

mining_df = pd.read_csv('mining_data.csv')
print(mining_df)
print()
print(mining_df.dtypes) # Check data types
print()
print(mining_df.isnull().sum()) # Check for missing values
print()
silica_concentrate_values = mining_df['% Silica Concentrate']
iron_concentrate_values = mining_df['% Iron Concentrate']
print(silica_concentrate_values.mean()) # Check for mean value of '% Silica Concentrate'
print()
print(iron_concentrate_values.max()) # Check for max value of '% Iron Concentrate'

# mining_df.hist(bins=30, figsize=(20, 20), color='forestgreen') # Histogram
# plt.show()

print(mining_df.corr())

# plt.figure(figsize=(8, 8))
# sns.heatmap(mining_df.corr(), annot=True)
# plt.show()

# plt.figure(figsize=(10, 10))
# plt.scatter(mining_df['% Silica Concentrate'], mining_df['% Iron Concentrate'])  # Scatter plot
# plt.xlabel('% Silica Concentrate')
# plt.ylabel('% Iron Concentrate')
# plt.title('Silica Concentrate vs Iron Concentrate')
# plt.show()

sns.scatterplot(x=mining_df['% Silica Concentrate'], y=mining_df['% Iron Concentrate'])
plt.show()  # Scatter plot

si_correlation_matrix = mining_df[['% Silica Concentrate', '% Iron Concentrate']].corr() # Correlation matrix
print("Correlation Matrix:") 
print(si_correlation_matrix)

sns.scatterplot(x= mining_df['% Iron Feed'], y= mining_df['% Silica Feed'])
plt.show()  # Scatter plot

is_correlation_matrix = mining_df[['% Iron Feed', '% Silica Feed']].corr() 
print("Correlation Matrix:") 
print(is_correlation_matrix)