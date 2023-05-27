

def split_rows(dataframe):
    """
    - Takes in one dataframe
    - Oversamples everything
    - Outputs one dataframe
    """
    new_data = []
    for index, row in dataframe.iterrows():
        type_value = row['type']
        text_value = row['text']
        if len(text_value) == 500:
            new_data.append([type_value, text_value, len(text_value)])

        if len(text_value) > 500:
            num_splits = len(text_value) // 500
            for i in range(num_splits):
                new_data.append([type_value, text_value[i * 500: (i + 1) * 500], len(text_value[i * 500: (i + 1) * 500])])

    dataframe = pd.DataFrame(new_data, columns=['type', 'text', 'length'])
    return dataframe

#calling and storing it
data = split_rows(data)



def split_targets1(dataframe):
    '''
    - Takes in one dataframe
    - Splits to 4 dataframes with personality type combinations
    - e_i_df, s_n_df, f_t_df, p_j_df
    '''
    e_i_df = dataframe[['text', 'type']].copy()
    e_i_df['type'] = e_i_df['type'].astype(str).str[0]

    s_n_df = dataframe[['text', 'type']].copy()
    s_n_df['type'] = s_n_df['type'].astype(str).str[1]

    f_t_df = dataframe[['text', 'type']].copy()
    f_t_df['type'] = f_t_df['type'].astype(str).str[2]

    p_j_df = dataframe[['text', 'type']].copy()
    p_j_df['type'] = p_j_df['type'].astype(str).str[3]

    return e_i_df, s_n_df, f_t_df, p_j_df

#calling and storing it into 4 dataframes
data_1, data_2, data_3, data_4 = split_targets1(data)



def binarize_and_adjust(dataframe):
    '''
    - Binarizes the "type" column
    - Calculates the lengths the binary values
    - Adjusts the number of rows to match the value with fewer rows
    - Spits out a dataframe
    '''
    # Binarize the "type" column
    dataframe['binary_type'] = dataframe['type'].apply(lambda x: 1 if x == dataframe['type'].iloc[0] else 0)

    # Calculate the length of each binary value
    length_1 = dataframe[dataframe['binary_type'] == 1].shape[0]
    length_0 = dataframe[dataframe['binary_type'] == 0].shape[0]

    # Determine the value with fewer rows
    min_length = min(length_1, length_0)

    # Adjust the number of rows to match the value with fewer rows
    dataframe_adjusted = pd.concat([dataframe[dataframe['binary_type'] == 1].head(min_length),
                             dataframe[dataframe['binary_type'] == 0].head(min_length)])

    dataframe_adjusted.drop(columns=['binary_type'], inplace=True)

    # Restore the original "type" values
    dataframe_adjusted['type'] = dataframe['type']

    return dataframe_adjusted

#calling and storing adjusted data into 4 dataframes
data_1_adjusted = binarize_and_adjust(data_1)
data_2_adjusted = binarize_and_adjust(data_2)
data_3_adjusted = binarize_and_adjust(data_3)
data_4_adjusted = binarize_and_adjust(data_4)
