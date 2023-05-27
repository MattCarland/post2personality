import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Remove URLs
def remove_urls(data):
    data['text'] = data['text'].apply(lambda x: re.sub(r'http\S+|www.\S+', '', x))
    return data['text']


# Remove social media handles
def remove_handles(data):
    data['text'] = data['text'].apply(lambda x: re.sub(r'@\w+', '', x))
    return data['text']


# Remove punctuation
def remove_punctuation(data):
    data['text'] = data['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    return data['text']


# Lowercase
def lower_case(data):
    data['text'] = data['text'].apply(lambda x: x.lower())
    return data['text']


# Remove special characters
def remove_special_characters(data):
    data['text'] = data['text'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s]+', '', x))
    return data['text']


# Remove white space
def remove_white_space(data):
    data['text'] = data['text'].apply(lambda x: x.strip())
    return data['text']


# Tokenizing
def tokenize(data):
    data['text'] = data['text'].apply(lambda x: word_tokenize(x))
    return data['text']

# if the toeknized list has more than 500 words
# it splits the list into another list of 500
# and drops the remaineder if less than 500 words
def split_rows(data):
    new_data = []
    for text in data['text']:
        if len(text) > 500:
            num_splits = len(text) // 500
            remainder = len(text) % 500
            for i in range(num_splits):
                new_data.append(text[i*500 : (i+1)*500])
            if remainder > 0:
                new_data.append(text[-remainder:])
        else:
            new_data.append(text)
    data['text'] = new_data
    return data['text']



# Stopword Removal
#stop_words = set(stopwords.words('english'))

def remove_stopwords(data):
    stop_words = set(stopwords.words('english'))
    data['text'] = data['text'].apply(lambda x: [w for w in x if w not in stop_words])
    return data['text']


# Text Lemmatization
def lemmatize(data):
    lemmatizer = WordNetLemmatizer()

    # Lemmatizing the verbs
    data['text'] = data['text'].apply(lambda x: [lemmatizer.lemmatize(word, pos="v") for word in x])

    # Lemmatizing the nouns
    data['text'] = data['text'].apply(lambda x: [lemmatizer.lemmatize(word, pos="n") for word in x])

    data['text'] = data['text'].apply(lambda x: ' '.join(x))
    return data['text']



def preprocessing(data):

    # remove URLs
    remove_urls(data)

    # remove social media handles
    remove_handles(data)

    # remove punctuation
    remove_punctuation(data)

    # lowercase
    lower_case(data)

    # remove special characters
    remove_special_characters(data)

    # remove white space
    remove_white_space(data)

    # tokenizing
    tokenize(data)

    # stopword removal
    remove_stopwords(data)

    # text lemmatization
    lemmatize(data)

    return data
