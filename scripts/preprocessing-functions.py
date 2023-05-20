import re
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')





#### Remove URLs
def remove_urls(data):
    data = re.sub(r'http\S+|www.\S+', '', data)
    return data


#### remove social media handles
def remove_handles(data):
    data = re.sub(r'@\w+', '', data)
    return data


#### remove punctuation
def punctuation(data):
    for punctuation in string.punctuation:
            data = data.replace(punctuation, '')
    return data


#### lowercase
def lower_case(data):
    return data.lower()


#### remove special characters
def remove_special_characters(data):
    data = re.sub(r'[^A-Za-z0-9\s]+', '', data)
    return data


#### remove white space
def white_space(data):
    return data.strip()


#### Tokenizing
def tokenize(data):
    data = word_tokenize(data)
    return data


#### Stopword Removal
stop_words = set(stopwords.words('english'))

def stopwords(data):
    data = [w for w in data if w not in stop_words]
    return data


#### Text Lemmatization
def lemmatize(data):

    # Lemmatizing the verbs
    data = [WordNetLemmatizer().lemmatize(word, pos = "v") for word in data]

    # Lemmatizing the nouns
    data = [WordNetLemmatizer().lemmatize(word, pos = "n") for word in data]

    return ' '.join(data)
