from nltk.tokenize import RegexpTokenizer

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
