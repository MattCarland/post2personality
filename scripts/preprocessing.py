from preprocessing-functions import *


def preprocessing(data):

    # remove URLs
    remove_urls(data)


    # remove social media handles
    remove_handles(data)


    # remove punctuation
    punctuation(data)


    # lowercase
    lower_case(data)


    # remove special characters
    remove_special_characters(data)


    # remove white space
    white_space(data)


    # tokenizing
    tokenize(data)


    # stopword removal
    stopwords(data)


    # text lemmatization
    lemmatize(data)


return data
