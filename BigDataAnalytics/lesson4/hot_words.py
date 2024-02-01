import os
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from collections import Counter
from nltk.util import ngrams

import pandas as pd

# To get stop words.
nltk.download('stopwords')


# -------------------------------------------------------------
# Create lower case array of words with no punctuation.
# -------------------------------------------------------------
def create_tokenized_array(sentences):
    # Initialize tokenizer and empty array to store modified sentences.
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_array = []
    for i in range(0, len(sentences)):
        # Convert sentence to lower case.
        sentence = sentences[i].lower()

        # Split sentence into array of words with no punctuation.
        words = tokenizer.tokenize(sentence)

        # Append word array to list.
        tokenized_array.append(words)

    print(tokenized_array)
    return tokenized_array  # send modified contents back to calling function.


# -------------------------------------------------------------
# Create array of words with no punctuation or stop words.
# -------------------------------------------------------------
def remove_stop_words(token_list):
    stop_words = set(stopwords.words('english'))
    shorter_sentences = []  # Declare empty array of sentences.

    for sentence in token_list:
        shorter_sentence = []  # Declare empty array of words in single sentence.
        for word in sentence:
            if word not in stop_words:

                # Remove leading and trailing spaces.
                word = word.strip()

                # Ignore single character words and digits.
                if len(word) > 1 and not word.isdigit():
                    # Add remaining words to list.
                    shorter_sentence.append(word)
        shorter_sentences.append(shorter_sentence)
    return shorter_sentences


# -------------------------------------------------------------
# Removes suffixes and rebuilds the sentences.
# -------------------------------------------------------------
def stem_words(sentence_arrays):
    ps = PorterStemmer()
    stemmed_sentences = []
    for sentence_array in sentence_arrays:
        stemmed_array = []  # Declare empty array of words.
        for word in sentence_array:
            stemmed_array.append(ps.stem(word))  # Add stemmed word.

        # Convert array back to sentence of stemmed words.
        delimiter = ' '
        sentence = delimiter.join(stemmed_array)

        # Append stemmed sentence to list of sentences.
        stemmed_sentences.append(sentence)
    return stemmed_sentences


# -------------------------------------------------------------
# Creates a matrix of word vectors.
# -------------------------------------------------------------
def vectorize_list(stemmed_list):
    # cv  = CountVectorizer(binary=True, ngram_range=(1, 4))
    cv = CountVectorizer(binary=True)

    cv.fit(stemmed_list)
    X = cv.transform(stemmed_list)
    print("\nNumber vector size: " + str(X.shape))
    return X


# -------------------------------------------------------------
# Build model and predict scores.
#
# Parameters:
# X         - X contains the stemmed and vectorized sentences.
# target    - The target is the known rating (0 to 4).

# Returns X_test, y_test, and y_predicted values.
# -------------------------------------------------------------
def model_and_predict(X, target):
    # Create training set with 75% of data and test set with 25% of data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, train_size=0.75
    )

    # Build the model with the training data.
    model = LogisticRegression(solver='newton-cg').fit(X_train, y_train)

    # Predict target values.
    y_prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_prediction)

    print("\n\n*** The accuracy score is: " + str(accuracy))

    print(classification_report(y_test, y_prediction))

    rmse2 = math.sqrt(metrics.mean_squared_error(y_test, y_prediction))
    print("RMSE: " + str(rmse2))

    # Your Python functions can return multiple values.
    return X_test, y_test, y_prediction


# Read in the file.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = f"{ROOT_PATH}\\..\\datasets\\"
CLEAN_DATA = "cleanedMovieReviews.tsv"

df = pd.read_csv(PATH + CLEAN_DATA, skiprows=1,
                 sep='\t', names=('PhraseId', 'SentenceId', 'Phrase', 'Sentiment'))

# Prepare the data.
df['PhraseAdjusted'] = create_tokenized_array(df['Phrase'])
df['PhraseAdjusted'] = remove_stop_words(df['PhraseAdjusted'])
df['PhraseAdjusted'] = stem_words(df['PhraseAdjusted'])
vectorizedList = vectorize_list(df['PhraseAdjusted'])

# Get predictions and scoring data.
# Target is the rating that we want to predict.
X_test, y_test, y_predicted = model_and_predict(vectorizedList, df['Sentiment'])


# Draw the confusion matrix.
def show_confusion_matrix(y_test, y_predicted):
    # You can print a simple confusion matrix with no formatting – this is easiest.
    cm = metrics.confusion_matrix(y_test.values, y_predicted)
    print(cm)


show_confusion_matrix(y_test, y_predicted)


def generate_word_list(word_df, score_start, score_end, n_gram_size):
    result_df = word_df[(word_df['Sentiment'] >= score_start) & \
                       (word_df['Sentiment'] <= score_end)]

    sentences = [sentence.split() for sentence in result_df['PhraseAdjusted']]
    word_array = []
    for i in range(0, len(sentences)):
        word_array += sentences[i]

    counter_list = Counter(ngrams(word_array, n_gram_size)).most_common(20)

    print("\n***N-Gram")
    for i in range(0, len(counter_list)):
        print("Occurrences: ", str(counter_list[i][1]), end=" ")
        delimiter = ' '
        print("N-Gram: ", delimiter.join(counter_list[i][0]))

    return counter_list


# Create two column matrix.
dfSub = df[['Sentiment', 'PhraseAdjusted']]
SCORE_RANGE_START = 4
SCORE_RANGE_END = 4
SIZE = 2
counter_list = generate_word_list(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, SIZE)

SIZE = 3
counter_list = generate_word_list(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, SIZE)
