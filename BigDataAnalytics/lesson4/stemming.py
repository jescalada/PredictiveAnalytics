import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

sentence1 = "Despite its fresh perspective, Banks's Charlie's Angels update " \
            + "fails to capture the energy or style that made it the beloved phenomenon it is."

sentence2 = "This 2019 Charlie's Angels is stupefyingly entertaining and " \
            + "hilarious. It is a stylish alternative to the current destructive blockbusters."

sentences = [sentence1, sentence2]


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


tokenized_list = create_tokenized_array(sentences)

# To get stop words.
nltk.download('stopwords')


# -------------------------------------------------------------
# Create array of words with no punctuation or stop words.
# -------------------------------------------------------------
def remove_stop_words(token_list, custom_stop_words=[]):
    stop_words = set(stopwords.words('english'))
    stop_words.update(custom_stop_words)

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
# Removes suffixes and rebuids the sentences.
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


sentence_arrays = remove_stop_words(tokenized_list, ['charlie', 'angels'])
print(sentence_arrays)

stemmed_sentences = stem_words(sentence_arrays)
print(stemmed_sentences)
