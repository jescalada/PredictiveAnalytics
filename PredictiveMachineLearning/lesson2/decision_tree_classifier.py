import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.tree import plot_tree

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CSV = "bill_authentication.csv"
dataset = pd.read_csv(PATH + CSV)
X = dataset.drop('Class', axis=1)
y = dataset['Class']
print(dataset)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=0)

classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


def show_accuracy_scores(y_test, y_pred):
    print("\nModel Evaluation")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("")
    tn = cm[0][0]
    fp = cm[0][1]
    tp = cm[1][1]
    fn = cm[1][0]
    accuracy = (tp + tn) / (tn + fp + tp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))


show_accuracy_scores(y_test, y_pred)

fig, ax = plt.subplots(figsize=(20, 10))

plot_tree(classifier.fit(X_train, y_train), max_depth=4, fontsize=4)
a = plot_tree(classifier,
              feature_names=['Variance', 'Skewness', 'Kurtosis', 'Entropy'],
              class_names=["I", "C"],
              filled=True,
              rounded=True,
              fontsize=14)
plt.show()
