# Load libraries
from sklearn.linear_model    import LogisticRegression
from sklearn                 import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

PATH = '/users/pm/desktop/daydocs/data/'
FILE = 'glass.csv'
df   = pd.read_csv(PATH + FILE)

# Show DataFrame contents.
print(df.head())

# Get X values and remove target column.
# Make copy to avoid over-writing.
X = df.copy()
del X['Type']

# Get y values
y = df['Type']
