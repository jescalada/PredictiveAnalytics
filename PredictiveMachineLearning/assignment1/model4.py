import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load training data from train.csv
pd.set_option('display.max_columns', None)
df = pd.read_csv('train.csv')

# Shuffle the data
df = df.sample(frac=1)

# Split the data into training and testing sets
X = df.drop('price_range', axis=1)
y = df['price_range']

# Keep only the most important features
X = X[['ram', 'battery_power', 'px_width', 'px_height', 'mobile_wt', 'talk_time', 'clock_speed']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Grab 50 samples from the test set to use for validation, then remove them from the test set
X_val = X_test[:50]
y_val = y_test[:50]
X_test = X_test[50:]
y_test = y_test[50:]

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Test the model
y_pred = model.predict(X_test_scaled)

# Print the results
print(classification_report(y_test, y_pred))

# Perform K-Fold Cross Validation
scores = cross_val_score(model, X_train_scaled, y_train, cv=10)
print('Cross-validated scores:', scores)
print('Mean score:', np.mean(scores))
print('Standard deviation of scores:', np.std(scores))

# Validate the model
X_val_scaled = scaler.transform(X_val)
y_val_pred = model.predict(X_val_scaled)
print(classification_report(y_val, y_val_pred))
print('F1 score:', f1_score(y_val, y_val_pred, average='weighted'))

# Show the confusion matrix for the test data
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
