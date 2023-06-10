
################################################################################
################################################################################
# # Key Parameters:
MODEL_NAME = 'bert-base-uncased'


################################################################################
################################################################################
# # Imports:
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import re
import numpy as np


################################################################################
################################################################################
# Import & load BERT model:
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)


################################################################################
################################################################################
# Target engineering:

def split_targets(dataframe):
    dataframe['e_i'] = dataframe['type'].astype(str).str[0]
    dataframe['s_n'] = dataframe['type'].astype(str).str[1]
    dataframe['f_t'] = dataframe['type'].astype(str).str[2]
    dataframe['p_j'] = dataframe['type'].astype(str).str[3]
    return dataframe


################################################################################
################################################################################
# Text-cleaning functions:

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


def remove_MBTI_types(dataframe):

    # Set list of terms (strings) to remove:
    masking = ['intj', 'intp', 'infj', 'infp',
               'istj', 'istp', 'isfj', 'isfp',
               'entj', 'entp', 'enfj', 'enfp',
               'estj', 'estp', 'esfj', 'esfp',

               'text', 'type']
    def remove_types(text):
        for type in masking:
            text = text.replace(type,'')
        return text
    # Update dataframe column with masked text:
    dataframe['text'] = dataframe['text'].apply(remove_types)
    return dataframe


def letters_only(dataframe):
    def strip_chars(text):
        return ''.join([char for char in text if char.isalpha()==True or char == ' '])
    dataframe['text'] = dataframe['text'].apply(strip_chars)
    return dataframe


################################################################################
################################################################################
# Master preprocessing functions:

def data_cleaning(data):
    '''
    This function performs basic text-cleaning to prepare for full preprocessing by BERT.
    '''
    data = lowercasing(data)
    data = remove_separators(data)
    data = remove_urls(data)
    data = remove_handles(data)
    data = remove_repeat_chars(data)
    data = remove_MBTI_types(data)
    # data = letters_only(data)
    data = remove_whitespace(data)

    print(f"cleaned dataset contains {data.shape[0]} rows and {data.shape[1]} columns")
    return data


def training_oversampling(data):
    """
    - Takes in one dataframe
    - Oversamples everything
    - Outputs one dataframe
    """
    new_data = []
    for index, row in data.iterrows():
        type_value = row['type']
        text_value = row['text']
        e_i = row['e_i']
        s_n = row['s_n']
        f_t = row['f_t']
        p_j = row['p_j']
        if len(text_value) <= 500:
            new_data.append([type_value, text_value, e_i, s_n, f_t, p_j])

        if len(text_value) > 500:
            num_splits = len(text_value) // 500
            for i in range(num_splits):
                new_data.append([type_value, text_value[i * 500: (i + 1) * 500], e_i, s_n, f_t, p_j])

    dataframe = pd.DataFrame(new_data, columns=['type', 'text', 'e_i', 's_n', 'f_t', 'p_j'])
    return dataframe


def BERT_vectorize(dataframe):
    samples = dataframe['text'].tolist()
    num_rows = len(samples)
    embeddings = []
    for i in range(num_rows):
        text = samples[i]

        # Tokenize the text
        encoded_inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']

        with torch.no_grad():
            # Pass input through the model
            outputs = model(input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state[0].tolist()  # Get the embedding for the input

        print(f'Processed {i+1}/{num_rows} samples')
        embeddings.append(embedding)
        # embeddings.append(np.asarray(embedding).astype(np.float32))

    dataframe['embeddings'] = embeddings
    return dataframe


def training_balancing(data):
    '''
    This function will take in the fully-"unpacked" (i.e. over-sampled) data,
    and will create FOUR new dataframes in which each class is perfectly bal-
    anced.
    '''
    e_i_data = data.drop(columns=['type', 's_n', 'f_t', 'p_j'])
    e_i_data.rename(columns={'e_i': 'type'}, inplace=True)
    e_i_data = e_i_data[['type', 'embeddings']].reset_index(drop=True)
    s_n_data = data.drop(columns=['type', 'e_i', 'f_t', 'p_j'])
    s_n_data.rename(columns={'s_n': 'type'}, inplace=True)
    s_n_data = s_n_data[['type', 'embeddings']].reset_index(drop=True)
    f_t_data = data.drop(columns=['type', 'e_i', 's_n', 'p_j'])
    f_t_data.rename(columns={'f_t': 'type'}, inplace=True)
    f_t_data = f_t_data[['type', 'embeddings']].reset_index(drop=True)
    p_j_data = data.drop(columns=['type', 'e_i', 's_n', 'f_t'])
    p_j_data.rename(columns={'p_j': 'type'}, inplace=True)
    p_j_data = p_j_data[['type', 'embeddings']].reset_index(drop=True)

    def balancer(dataframe):
        '''
        - Binarizes the "type" column
        - Calculates the lengths the binary values
        - Adjusts the number of rows to match the value with fewer rows
        - Spits out a dataframe
        '''
        # Binarize the "type" column
        dataframe['binary_type'] = dataframe['type'].apply(lambda x: 1 if x == dataframe['type'].iloc[0] else 0)

        # Calculate the length of each binary value
        # length_1 = dataframe[dataframe['binary_type'] == 1].shape[0] #######################################################################
        # length_0 = dataframe[dataframe['binary_type'] == 0].shape[0] #######################################################################
        values = dataframe['type'].value_counts()

        # Determine the value with fewer rows
        # min_length = min(length_1, length_0) ###############################################################################################
        min_length = min(values)

        # Adjust the number of rows to match the value with fewer rows
        dataframe_adjusted = pd.concat([dataframe[dataframe['binary_type'] == 1].head(min_length),
                                dataframe[dataframe['binary_type'] == 0].head(min_length)])

        dataframe_adjusted.drop(columns=['binary_type'], inplace=True)

        # Restore the original "type" values
        dataframe_adjusted['type'] = dataframe['type']

        return dataframe_adjusted.reset_index(drop=True)

    df_dict = {'e_i': balancer(e_i_data), 's_n': balancer(s_n_data), 'f_t': balancer(f_t_data), 'p_j': balancer(p_j_data)}
    return df_dict


def reform(dataframe_dict):
    '''
    Takes in a DICT of dataframes and reforms into a LIST of dataframes.

    (This is just to keep the format of the final output consistent with that
    of the original v1 preprocessing pipeline; it's not functionally important)
    '''
    e_i_data = dataframe_dict['e_i']
    s_n_data = dataframe_dict['s_n']
    f_t_data = dataframe_dict['f_t']
    p_j_data = dataframe_dict['p_j']

    return [e_i_data, s_n_data, f_t_data, p_j_data]


################################################################################
################################################################################
# TRAINING DATA Pipeline:

def bert_training_preprocessing(input_dataframe):
    training_data = split_targets(input_dataframe)
    cleaned_data = data_cleaning(training_data)
    oversampled_data = training_oversampling(cleaned_data)
    vectorized_data = BERT_vectorize(oversampled_data)
    balanced_data = training_balancing(vectorized_data)
    preprocessed_data = reform(balanced_data)

    return preprocessed_data


################################################################################
################################################################################
# PREDICTION DATA Pipeline:

def bert_prediction_preprocessing(data):
    cleaned_data = data_cleaning(data)
    vectorized_data = BERT_vectorize(cleaned_data)[['embeddings']]

    return [vectorized_data, vectorized_data, vectorized_data, vectorized_data]
