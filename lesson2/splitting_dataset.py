import pandas as pd
from sklearn.model_selection import train_test_split

# Create DataFrame.
dataSet = {'days': [0.2, 0.32, 0.38, 0.41, 0.43],
           'growth': [0.1, 0.15, 0.4, 0.6, 0.44] }
df = pd.DataFrame(dataSet, columns=['days', 'growth'])

# Store x and y values.
X = df['days']
target = df['growth']

# Create training set with 60% of data and test set with 40% of data.
X_train, X_test, y_train, y_test = train_test_split(
    X, target, train_size=0.8
)

print(f"X_train:\n{X_train}\nX_test: \n{X_test}\ny_train: \n{y_train}\ny_test: \n{y_test}")
