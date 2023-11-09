from mlxtend.data import loadlocal_mnist
from sklearn.metrics import classification_report
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "\\..\\datasets\\"

X, y = loadlocal_mnist(
    images_path=f"{ROOT_PATH}{DATASET_DIR}t10k-images.idx3-ubyte",
    labels_path=f"{ROOT_PATH}{DATASET_DIR}t10k-labels.idx1-ubyte"
)

# http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/
print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\n1st row', X[0])

# Split the data.
from sklearn.model_selection import train_test_split

# test_size: what proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split(X,
                                                            y,
                                                            test_size=1 / 7.0,
                                                            random_state=0)
# Show image.
print("Image size: ")
print(train_img[0].shape)

import matplotlib.pyplot as plt
import numpy as np

first_image = train_img[0]
first_image = np.array(first_image, dtype='float')
print(len(first_image))
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

second_image = train_img[1]
second_image = np.array(second_image, dtype='float')
pixels = second_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

third_image = train_img[2]
third_image = np.array(third_image, dtype='float')
pixels = third_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

from sklearn.decomposition import PCA

# Make an instance of the PCA.
pca = PCA(.95)
pca.fit(train_img)

# Data is transformed with PCA.
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver='lbfgs', max_iter=1000)
logisticRegr.fit(train_img, train_lbl)
y_pred = logisticRegr.predict(test_img)
score = logisticRegr.score(test_img, test_lbl)
print(score)

# Show confusion matrix and accuracy scores.
import pandas as pd

cm = pd.crosstab(test_lbl, y_pred, rownames=['Actual'],
                 colnames=['Predicted'])

print("\n*** Confusion Matrix")
print(cm)

print("\n*** Classification Report")
print(classification_report(test_lbl, y_pred))
