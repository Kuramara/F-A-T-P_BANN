import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('EXPORT_2015_monthly_annual.csv')

# Convert 'FREE_ON_BOARD' to numeric
df['FREE_ON_BOARD'] = df['FREE_ON_BOARD'].str.replace(',', '').astype(float)

# Group by commodity description and sum the 'FREE_ON_BOARD' values
grouped = df.groupby('COMMODITY_DESCRIPTION')['FREE_ON_BOARD'].sum().reset_index()

# Sort by the total 'FREE_ON_BOARD' value
sorted_df = grouped.sort_values('FREE_ON_BOARD', ascending=False)

# Get the top 10
top_10 = sorted_df.head(10)

# Print the result
print(top_10)
