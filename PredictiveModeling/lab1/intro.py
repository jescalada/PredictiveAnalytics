import pandas as pd

# Create data set.
dataSet = {'First Name': ['Jonny', 'Holly', 'Nira'],
           'Last Name': ['Staub', 'Conway', 'Arora'],
           'Grade': [85,95,91] }

# Create dataframe with data set and named columns.
# Column names must match the dataSet properties.
df = pd.DataFrame(dataSet, columns=['First Name', 'Last Name', 'Grade'])

# Show DataFrame
print(df)


# Create data set.
market_dataset = {'Market': ['S&P 500', 'Dow', 'Nikkei'],
           'Last': [2932.05, 26485.01, 21087.16] }

# Create dataframe with data set and named columns.
df2 = pd.DataFrame(market_dataset, columns= ['Market', 'Last'])

# Show original DataFrame.
print("\n*** Original DataFrame ***")
print(df2)

# Create change column.
change = [-21.51, -98.41, -453.83]

# Append change columns.
df2['Change'] = change

# Compute percentage change.
df2['Percentage Change'] = round(df2['Change'] / df2['Last'], 4) * 100

# Show revised DataFrame.
print("\n*** Adjusted DataFrame ***")
print(df2)

for i in range(len(df2)):
    print(df2.loc[i]['Last'])

print(df2.loc[0]['Market'], end=" ")
print(df2.loc[0]['Last'], end=" ")
print(df2.loc[0]['Change'], end=" ")


# Adding two dataframes
dataSet = {'Market': ['S&P 500', 'Dow', 'Nikkei'],
           'Last': [2932.05, 26485.01, 21087.16] }

# Create dataframe with data set and named columns.
df1 = pd.DataFrame(dataSet, columns= ['Market', 'Last'])

# Show original DataFrame.
print("\n*** Original DataFrame ***")
print(df1)


dataSet2 = { 'Market': ['Hang Seng', 'DAX'],
             'Last': [26918.58, 11872.44]}
df2 = pd.DataFrame(dataSet2, columns= ['Market', 'Last'])

df1 = df1._append(df2)
print("\n*** Adjusted DataFrame ***")
print(df1)

dataSet3 = {'Market': ['FTSE100'],
            'Last': [7407.06]}

df3 = pd.DataFrame(dataSet3, columns= ['Market', 'Last'])
df3 = df1._append(df3)

print("\n*** Adjusted DataFrame ***")
print(df3)


# Create data set.
dataSet = {'Market': ['S&P 500', 'Dow', ],
           'Last': [2932.05, 26485.01 ]}

# The dictionary is an object made of name value pairs.
stockDictionary = {'Market': 'Nikkei', 'Last': 21087.16 }

# Create dataframe with data set and named columns.
df = pd.DataFrame(dataSet, columns= ['Market', 'Last'])

# Show original DataFrame.
print("\n*** Original DataFrame ***")

df = df._append(stockDictionary, ignore_index=True)
print(df)

new_stock = {'Market': 'DAX', 'Last': 11872.44 }
df = df._append(new_stock, ignore_index=True)

print("\n*** Adjusted DataFrame ***")
print(df)
