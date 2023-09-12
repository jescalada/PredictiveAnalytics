import pandas as pd

# Import data into a DataFrame.
path = "C:\\Users\\juane\\Desktop\\Juan\\Predictive\\bodyfat.txt"

df = pd.read_table(path, skiprows=1,
                   delim_whitespace=True,
                   names=('Density', 'Pct.BF', 'Age', 'Weight', 'Height', 'Neck',
                          'Chest', 'Abdomen', 'Waist', 'Hip', 'Thigh',
                          'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist'))

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print("\n FIRST 2 ROWS") # Prints title with space before.
print(df.head(2))

print("\n LAST 2 ROWS")
print(df.tail(2))

# Show data types for each columns.
print("\n DATA TYPES") # Prints title with space before.
print(df.dtypes)

# Show statistical summaries for numeric columns.
print("\nSTATISTIC SUMMARIES for NUMERIC COLUMNS")
print(df.describe().round(2))

# Show table with Height, Waist, Weight, and Pct.BF columns.
df_small = df[['Height', 'Waist', 'Weight', 'Pct.BF']]

print("\n SMALL TABLE WITH HEIGHT, WAIST, WEIGHT, AND PCT.BF COLUMNS")
print(df_small.head(4))

df_small.rename({'Pct.BF': 'Percent Body Fat'}, axis=1)
print(df_small.head(4))


# Import data into a DataFrame.
path = "C:\\Users\\juane\\Desktop\\Juan\\Predictive\\babysamp-98.txt"
df = pd.read_csv(path, skiprows=1,
                   sep='\t',
                   names=('MomAge', 'DadAge', 'MomEduc', 'MomMarital', 'numlive',
                          "dobmm", 'gestation', 'sex', 'weight', 'prenatalstart',
                          'orig.id', 'preemie'))

# Rename the columns so they are more reader-friendly.
df = df.rename({'MomAge': 'Mom Age', 'DadAge':'Dad Age',
                'MomEduc':'Mom Edu', 'weight':'Weight'}, axis=1)  # new method
# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)
print(df.head())


print("\nTOP FREQUENCY FIRST")
print(df['Mom Edu'].value_counts())

print("\nLOWEST FREQUENCY FIRST")
print(df['Mom Edu'].value_counts(ascending=True))

print("\nFREQUENCY SORTED by MOTHER EDUCATION LEVEL")
print(df['Mom Edu'].value_counts().sort_index())


# Import data into a DataFrame.
df = pd.read_csv(path, skiprows=1,
                   sep='\t',
                   names=('MomAge', 'DadAge', 'MomEduc', 'MomMarital', 'numlive',
                          "dobmm", 'gestation', 'sex', 'weight', 'prenatalstart',
                          'orig.id', 'preemie'))

# Rename the columns so they are more reader-friendly.
df = df.rename({'MomAge': 'Mom Age', 'DadAge':'Dad Age',
                'MomEduc':'Mom Edu', 'weight':'Weight'}, axis=1)  # new method
# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

# Sort by ascending gestation time and then by ascending weight.
dfSorted = df.sort_values(['gestation', 'Weight'], ascending=[True, True])
print(dfSorted)


# Exercise 16

# Import data into a DataFrame.
df = pd.read_csv(path, skiprows=1,
                   sep='\t',
                   names=('MomAge', 'DadAge', 'MomEduc', 'MomMarital', 'numlive',
                          "dobmm", 'gestation', 'sex', 'weight', 'prenatalstart',
                          'orig.id', 'preemie'))

# Rename the columns so they are more reader-friendly.
df = df.rename({'MomAge': 'Mom Age', 'DadAge':'Dad Age',
                'MomEduc':'Mom Edu', 'weight':'Weight'}, axis=1)  # new method
# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)


print("Mom Age Summary Statistics")
print(f"Count: {df['Mom Age'].count()}")
print(f"Min: {df['Mom Age'].min()}")
print(f"Max: {df['Mom Age'].max()}")
print(f"Mean: {df['Mom Age'].mean()}")
print(f"Median: {df['Mom Age'].median()}")
print(f"Standard Deviation: {df['Mom Age'].std()}")

print("---------------------------------")
# Exercise 17

# The data file path and file name need to be configured.
PATH = "C:\\Users\\juane\\Desktop\\Juan\\Predictive\\"
CSV_DATA = "phone_data.csv"

# Note this has a comma separator.
df = pd.read_csv(PATH + CSV_DATA, skiprows=1, encoding="ISO-8859-1", sep=',',
                 names=('index', 'date', 'duration', 'item', 'month', 'network',
                        'network_type'))
# Get count of items per month.
dfStats = df.groupby('network')['index'] \
    .count().reset_index().rename(columns={'index': '# Calls'})

# Get duration mean for network groups and convert to DataFrame.
dfDurationMean = df.groupby('network')['duration'] \
    .mean().reset_index().rename(columns={'duration': 'Duration Mean'})

# Get duration max for network groups and convert to DataFrame.
dfDurationMax = df.groupby('network')['duration'] \
    .max().reset_index().rename(columns={'duration': 'Duration Max'})

# Get duration minimum for network groups and convert to DataFrame.
dfDurationMin = df.groupby('network')['duration'] \
    .min().reset_index().rename(columns={'duration': 'Duration Min'})

# Get duration SD for network groups and convert to DataFrame.
dfDurationStd = df.groupby('network')['duration'] \
    .std().reset_index().rename(columns={'duration': 'Duration SD'})

# Append duration mean to stats matrix.
dfStats['Duration Mean'] = dfDurationMean['Duration Mean']

# Append duration max to stats matrix.
dfStats['Duration Max'] = dfDurationMax['Duration Max']

# Append duration min to stats matrix.
dfStats['Duration Min'] = dfDurationMin['Duration Min']

# Append duration SD to stats matrix.
dfStats['Duration SD'] = dfDurationStd['Duration SD']
print(dfStats)

# Exercise 18

print("---------------------------------")

# Note this has a comma separator.
df = pd.read_csv(PATH + CSV_DATA, skiprows=1, encoding="ISO-8859-1", sep=',',
                 names=('index', 'date', 'duration', 'item', 'month', 'network',
                        'network_type'))
# Get count of items per month.
dfStats = df.groupby('network_type')['index'] \
    .count().reset_index().rename(columns={'index': '# Calls'})

# Get duration mean for network groups and convert to DataFrame.
dfDurationMean = df.groupby('network_type')['duration'] \
    .mean().reset_index().rename(columns={'duration': 'Duration Mean'})

# Get duration max for network groups and convert to DataFrame.
dfDurationMax = df.groupby('network_type')['duration'] \
    .max().reset_index().rename(columns={'duration': 'Duration Max'})

# Get duration minimum for network groups and convert to DataFrame.
dfDurationMin = df.groupby('network_type')['duration'] \
    .min().reset_index().rename(columns={'duration': 'Duration Min'})

# Get duration SD for network groups and convert to DataFrame.
dfDurationStd = df.groupby('network_type')['duration'] \
    .std().reset_index().rename(columns={'duration': 'Duration SD'})

# Append duration mean to stats matrix.
dfStats['Duration Mean'] = dfDurationMean['Duration Mean']

# Append duration max to stats matrix.
dfStats['Duration Max'] = dfDurationMax['Duration Max']

# Append duration min to stats matrix.
dfStats['Duration Min'] = dfDurationMin['Duration Min']

# Append duration SD to stats matrix.
dfStats['Duration SD'] = dfDurationStd['Duration SD']
print(dfStats)


# Exercise 19
df = pd.read_csv(path, skiprows=1,
                   sep='\t',
                   names=('MomAge', 'DadAge', 'MomEduc', 'MomMarital', 'numlive',
                          "dobmm", 'gestation', 'sex', 'weight', 'prenatalstart',
                          'orig.id', 'preemie'))

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

dfStats = df.groupby('sex')['orig.id'].count().reset_index().rename(columns={'orig.id': '# Babies'})

dfWeightMax = df.groupby('sex')['weight'].max().reset_index().rename(columns={'weight': 'Weight Max'})
dfWeightMin = df.groupby('sex')['weight'].min().reset_index().rename(columns={'weight': 'Weight Min'})
dfWeightMean = df.groupby('sex')['weight'].mean().reset_index().rename(columns={'weight': 'Weight Mean'})

dfStats['Weight Max'] = dfWeightMax['Weight Max']
dfStats['Weight Min'] = dfWeightMin['Weight Min']
dfStats['Weight Mean'] = dfWeightMean['Weight Mean']

print(dfStats)

print("---------------------------------")
# Exercise 20

# Create data set.
dataSet = { 'Fahrenheit': [85,95,91] }

# Create dataframe with data set and named columns.
# Column names must match the dataSet properties.
df = pd.DataFrame(dataSet, columns= ['Fahrenheit'])

df['Celsius'] = (df['Fahrenheit']-32)*5/9
df['Kelvin'] = df['Celsius'] + 273.15

# Show DataFrame
print(df)

print("---------------------------------")
# Exercise 21

df = pd.read_csv(PATH + CSV_DATA, skiprows=1,  encoding = "ISO-8859-1", sep=',',
                 names=('index', 'date', 'duration', 'item', 'month','network',
                        'network_type' ))

df2 = df.groupby(['network_type','item'])['duration'].mean().reset_index()
print(df2)
