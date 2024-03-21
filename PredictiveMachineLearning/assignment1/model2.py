import pandas as pd

# Load training data from train.csv
pd.set_option('display.max_columns', None)
df = pd.read_csv('train.csv')

# Make an ensemble model using mlextend.regressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

# Split the data into training and testing sets
X = df.drop('price_range', axis=1)
y = df['price_range']

# Keep only the most important features: 'ram', 'battery_power', 'px_width', 'px_height', 'mobile_wt', 'talk_time'
X = X[['ram', 'battery_power', 'px_width', 'px_height']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Grab a validation set of 50 samples
X_val = X_test[:50]
y_val = y_test[:50]
X_test = X_test[50:]
y_test = y_test[50:]

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the models
model1 = LogisticRegression()
model2 = RandomForestClassifier(n_estimators=100)
model3 = BaggingClassifier(n_estimators=100)
model4 = VotingClassifier(estimators=[('lr', model1), ('rf', model2), ('bc', model3)], voting='hard')

# Train the models
model1.fit(X_train_scaled, y_train)
model2.fit(X_train_scaled, y_train)
model3.fit(X_train_scaled, y_train)
model4.fit(X_train_scaled, y_train)

# Build the ensemble model
y_pred = model4.predict(X_test_scaled)

# Print the results
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Perform K-Fold Cross Validation

from sklearn.model_selection import cross_val_score
import numpy as np

# Perform 10-fold cross validation
scores = cross_val_score(model4, X_train, y_train, cv=10)
print('Cross-validated scores:', scores)
print('Mean score:', np.mean(scores))
print('Standard deviation of scores:', np.std(scores))

# Validate the model
X_val_scaled = scaler.transform(X_val)
y_val_pred = model4.predict(X_val_scaled)
print(classification_report(y_val, y_val_pred))

# Print F1 score
from sklearn.metrics import f1_score
print('F1 score:', f1_score(y_val, y_val_pred, average='weighted'))