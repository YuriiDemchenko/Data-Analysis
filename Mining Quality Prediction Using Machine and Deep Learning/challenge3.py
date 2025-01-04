import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

mining_df = pd.read_csv('mining_data.csv')
sns.scatterplot(x=mining_df['% Silica Concentrate'], y=mining_df['% Iron Concentrate'])
plt.show()  # Scatter plot

si_correlation_matrix = mining_df[['% Silica Concentrate', '% Iron Concentrate']].corr() # Correlation matrix
print("Correlation Matrix:") 
print(si_correlation_matrix)
print()

sns.scatterplot(x= mining_df['% Iron Feed'], y= mining_df['% Silica Feed'])
plt.show()  # Scatter plot

is_correlation_matrix = mining_df[['% Iron Feed', '% Silica Feed']].corr()  # Correlation matrix
print("Correlation Matrix:") 
print(is_correlation_matrix)