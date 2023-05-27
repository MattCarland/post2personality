################################################################################
################################################################################
# Global parameters to set:

DATASET_NUMBER = 0   # <-- (1-3, or '0' for ALL data)

MIN_DOC_FREQ = 0.02
MAX_DOC_FREQ = 0.80

#__________________________________
# For testing & diagnostics only:
DATA_LIMIT = 0  # <-- use this parameter to limit the number of rows processed; '0' will process all data


################################################################################
################################################################################
# Imports:
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
import string
import re
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer # <-- I went with a TF-IDF approach for now, but we could switch to this
                                                            #      later on if we want to try a simpler "bag-of-words" approach.


################################################################################
################################################################################
# Data intake:
#   Note: should eventually be replaced with its own dedicated script, esp.
#         one google cloud storage is up and running.

### Use ALL data:
if DATASET_NUMBER == 0:

    data_1 = pd.read_csv('data/csv/MBTI 500.csv')
    data_1 = data_1[['type', 'posts']]
    data_1.rename(columns={'posts': 'text'}, inplace=True)

    data_2 = pd.read_csv('data/csv/twitter_MBTI.csv')
    data_2 = data_2[['label', 'text']]
    data_2.rename(columns={'label': 'type'}, inplace=True)

    data_3 = pd.read_csv('data/csv/mbti_1.csv')
    data_3.rename(columns={'posts': 'text'}, inplace=True)

    dataframes = [data_1, data_2, data_3]
    data = pd.concat(dataframes)

# 1) Types-500 dataset (already preprocessed):
if DATASET_NUMBER == 1:
    data = pd.read_csv('data/csv/MBTI 500.csv')
    data = data[['type', 'posts']]
    data.rename(columns={'posts': 'text'}, inplace=True)

# 2) Twitter dataset:
if DATASET_NUMBER == 2:
    data = pd.read_csv('data/csv/twitter_MBTI.csv')
    data = data[['label', 'text']]
    data.rename(columns={'label': 'type'}, inplace=True)

# 3) PersonalityCafe forums dataset:
if DATASET_NUMBER == 3:
    data = pd.read_csv('data/csv/mbti_1.csv')
    data.rename(columns={'posts': 'text'}, inplace=True)


print(f'\nDataset contains {data.shape[0]} rows\n')

#_________________________________________
### OPTIONAL: DATA LIMITING
if DATA_LIMIT > 0:
    data = data.iloc[:DATA_LIMIT]
#_________________________________________


################################################################################
################################################################################
# Support functions:
  # Note: these are necessary for feature extraction/feature engineering code

def is_english(text: str) -> bool:
    '''
    Code for checking if text sample contains non-English characters. Takes in
    a STRING and returns a BOOL.
    '''
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def split_into_sentences(text: str) -> list[str]:
    """
    [REQUIRES: RegEx library import]
    -----------
    Split a text sample into individual sentences for further analyses.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.
    """
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|edu|me)"
    digits = "([0-9])"
    multiple_dots = r'\.{2,}'
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences


def get_avg_sentence_lengths(text: str) -> float:
    '''
    [REQUIRES: NumPy & 'split_into_sentences()' support function.]
    -----------
    This takes in each text sample as a single STRING, and returns a FLOAT
    representing the average length of each sentence in a given text sample.
    (Note: code ignores any "sentences" which are only one "word" long, as
    these are most often subtle parsing errors.)
    '''
    sentences = split_into_sentences(text)
    sentence_lengths = []
    for sentence in sentences:
        sentence_stripped = ''.join([char for char in sentence if char.isalpha()==True or char==' ' or char=='-'])
        sentence_lengths.append(len(sentence_stripped.split()))
    return np.mean([length for length in sentence_lengths if length > 1])


def get_avg_word_lengths(text: str) -> float:
    '''
    [REQUIRES: NumPy & 'string' library import]
    -----------
    This takes in each text sample as a single STRING, and returns a FLOAT
    representing the average word length used within each text sample.
    '''
    for punc in string.punctuation:
        text = text.replace(punc, ' ')
    words = ''.join([char for char in text if char.isalpha()==True or char==' ' or char=='-']).strip().split()
    return (np.mean([len(word) for word in words]))


def type_to_token_ratio(tokenized_text: list[str]) -> float:
    '''
    Takes in a list of TOKENIZED text, and returns a FLOAT representing each
    samples' "type-to-token" ratio, which is a measure of vocabulary uniqueness.
    Higher scores are indicative of a broader vocabulary, whereas relatively
    lower scores indicate more repetitive patterns of word usage.
    '''
    total_words = len(tokenized_text)
    unique_words = list(set(tokenized_text))
    return ((len(unique_words)) / total_words)


################################################################################
################################################################################
# Core preprocessing functions:

def lowercasing(dataframe):
    def lower_case(text):
        return text.lower()
    dataframe['text'] = dataframe['text'].apply(lower_case)
    return dataframe


def remove_separators(dataframe):
    def remove_bars(text):
        # We want to remove post separators ('|||') & replace w/ blank space:
        return text.replace('|||', ' ')
    dataframe['text'] = dataframe['text'].apply(remove_bars)
    return dataframe


def remove_urls(dataframe):
    def URL_removal(text):
        return re.sub(r'http\S+|www.\S+', '', text)
    dataframe['text'] = dataframe['text'].apply(URL_removal)
    return dataframe


def remove_handles(dataframe):
    def handle_remover(text):
        return re.sub(r'@\w+', '', text)
    dataframe['text'] = dataframe['text'].apply(handle_remover)
    return dataframe


def remove_punctuation(dataframe):
    # NOTE: Possible issues: will destroy emoticons (':-P', ':-(', ':D' ), etc.
    #         can find a better work-around for this later
    def punctuation_remover(text):
        return ''.join([char if char not in string.punctuation else ' ' for char in str(text)])
    dataframe['text'] = dataframe['text'].apply(punctuation_remover)
    return dataframe


def remove_MBTI_types(dataframe):
    # Set list of terms (strings) to remove:
    masking = ['intj', 'intp', 'infj', 'infp',
               'istj', 'istp', 'isfj', 'isfp',
               'entj', 'entp', 'enfj', 'enfp',
               'estj', 'estp', 'esfj', 'esfp']
    def remove_types(text):
        for type in masking:
            text = text.replace(type,'')
        return text
    # Update dataframe column with masked text:
    dataframe['text'] = dataframe['text'].apply(remove_types)
    return dataframe


def remove_repeat_chars(dataframe):
    '''Takes any characters repeated more than 2 times in a row, and reduces
    them down to just two repetitions.'''
    def remove_repeats(text):
        return re.sub(r'(\w)\1+', r'\1\1', text)
    dataframe['text'] = dataframe['text'].apply(remove_repeats)
    return dataframe


def remove_whitespace(dataframe):
    def white_space(text):
        return ' '.join(text.split()).strip()
    dataframe['text'] = dataframe['text'].apply(white_space)
    return dataframe


def letters_only(dataframe):
    def strip_chars(text):
        return ''.join([char for char in text if char.isalpha()==True or char == ' '])
    dataframe['text'] = dataframe['text'].apply(strip_chars)
    return dataframe


def tokenize(dataframe):
    dataframe['text'] = dataframe['text'].apply(lambda row: nltk.word_tokenize(row))
    return dataframe


def remove_stopwords(dataframe):
    # Initialize list of stopwords to use:
    stop_words = set(stopwords.words('english'))
    # Remove stopwords:
    dataframe['text'] = dataframe['text'].apply(lambda x: [word for word in x if word not in stop_words])
    return dataframe


def lemmatize_data(dataframe):
    def lemmatizer(text):
        # Lemmatizing the verbs:
        text = [WordNetLemmatizer().lemmatize(word, pos = "v") for word in text]
        # Lemmatizing the nouns:
        text = [WordNetLemmatizer().lemmatize(word, pos = "n") for word in text]
        return text
    dataframe['text'] = dataframe['text'].apply(lemmatizer)
    return dataframe


def correct_spelling(dataframe):
    # NOTE: This function is highly costly in time.
    for i in range(dataframe.shape[0]):
        dataframe.text[i] = str(TextBlob(str(' '.join(dataframe.iloc[i].text))).correct()).split()
    return dataframe


def vectorize(dataframe):
    # Instantiate vectorizer:
    vectorizer = TfidfVectorizer(min_df=MIN_DOC_FREQ, max_df=MAX_DOC_FREQ)
    # Fit & transform training data:
    X = vectorizer.fit_transform(dataframe['text'].apply(' '.join))
    # Re-cast vectorized data into DataFrame format:
    X_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    # Append vectorized output onto the input dataframe:
    return pd.concat([dataframe, X_df], axis=1)


################################################################################
################################################################################
# Feature engineering (in progress; more to come):

def lowercase_targets(dataframe):
    dataframe['type'] = dataframe['type'].str.lower()
    return dataframe


def split_targets(dataframe):
    dataframe['e_i'] = dataframe['type'].astype(str).str[0]
    dataframe['s_n'] = dataframe['type'].astype(str).str[1]
    dataframe['f_t'] = dataframe['type'].astype(str).str[2]
    dataframe['p_j'] = dataframe['type'].astype(str).str[3]
    return dataframe


def get_type_to_token_ratio(dataframe):
    dataframe['type_to_token_ratio'] = dataframe['text'].apply(type_to_token_ratio)
    return dataframe


def get_avg_word_length(dataframe):
    dataframe['avg_word_length'] = dataframe['text'].apply(get_avg_word_lengths)
    return dataframe


def get_avg_sentence_length(dataframe):
    dataframe['avg_sentence_length'] = dataframe['text'].apply(get_avg_sentence_lengths)
    return dataframe


################################################################################
################################################################################
# Main pipeline for TRAINING (labeled) data:

def training_preprocessing(data):
    '''
    Run this function to clean data for model TRAINING.

    (This code assumes that the target label column ('type') is present.)
    '''
    data = lowercase_targets(data)
    data = split_targets(data)

    # Text cleaning, etc. (all input/output == STR format)
    data = lowercasing(data)
    data = remove_separators(data)
    data = remove_urls(data)
    data = remove_handles(data)
    data = remove_punctuation(data)
    data = remove_MBTI_types(data)
    data = remove_repeat_chars(data)
    data = remove_whitespace(data)
    data = get_avg_word_length(data)
    data = letters_only(data)

    # Post-tokenization: all input/output == LIST format):
    data = tokenize(data)
    data = remove_stopwords(data)
    data = lemmatize_data(data)
    data = get_type_to_token_ratio(data)
    # [[TBD: "Oversampling" method (chopping up samples into multiple 500-word units) will go here]]
    # data = correct_spelling(data) # Spelling correction step -- WARNING: VERY TIME-CONSUMING, only use for final production models.

    # Final vectorization:
    data = vectorize(data)

    print(f"final dataset contains {data.shape[0]} rows and {data.shape[1]} columns")

    return data


################################################################################
################################################################################
# Pipeline for NEW data (for PREDICTIONS):

def prediction_preprocessing(data):
    '''
    Use this function to clean USER data, for feeding into the production model
    in order to generate PREDICTIONS.

    (This code skips over any steps that require the label column ('type') to be
    present.)
    '''

    # Text cleaning, etc. (all input/output == STR format)
    data = lowercasing(data)
    data = remove_separators(data)
    data = remove_urls(data)
    data = remove_handles(data)
    data = remove_punctuation(data)
    data = remove_MBTI_types(data)
    data = remove_repeat_chars(data)
    data = remove_whitespace(data)
    data = get_avg_word_length(data)
    data = letters_only(data)

    # Post-tokenization: all input/output == LIST format):
    data = tokenize(data)
    data = remove_stopwords(data)
    data = lemmatize_data(data)
    data = get_type_to_token_ratio(data)
    # data = correct_spelling(data) # Spelling correction step -- WARNING: VERY TIME-CONSUMING, only use for final production models.

    # Final vectorization:
    data = vectorize(data)

    return data
