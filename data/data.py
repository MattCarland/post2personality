import os
import pandas as pd

class Data:
    def get_all_data(self):
        """
        This function returns a Python dict.
        Its keys should be 'set01', 'set02', 'set03' etc...
        Its values should be pandas.DataFrames loaded from csv files
        """

        csv_path = "./data/csv/"

        file_names = [f for f in os.listdir(csv_path) if f.endswith(".csv")]

        key_names = [
            key_name.replace(".csv", "")
            for key_name in file_names
        ]
        # Create the dictionary
        data = {}
        for k, f in zip(key_names, file_names):
            data[k] = pd.read_csv(os.path.join(csv_path, f))
        return data
        # print(data)

    def get_set01(self):
        """
        This function returns data_mbti_1 DataFrame
        """
        return self.get_all_data()["data_mbti_1"]

    def get_set02(self):
        """
        This function returns data_MBTI_500 DataFrame
        """
        return self.get_all_data()["data_MBTI_500"]

    def get_set03(self):
        """
        This function returns a data_twitter_MBTI DataFrame
        """
        return self.get_all_data()["data_twitter_MBTI"]
