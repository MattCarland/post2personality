# Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict
import pickle

def initialize_model():
    model = SGDClassifier()
    return model



def PredictDict(model, X_test, y_test):
    y_pred = cross_val_predict(model, X_test, y_test, cv = 10)
    report = classification_report(y_test, y_pred, output_dict=True)
    bal_acc = balanced_accuracy_score(y_pred, y_test)
    report['bal_acc'] = bal_acc
    return report



def train_model(data_list,
                 model_select,
                 random_state = 1,
                 Prediction = True):
    """
        Test a machine learning model on a given dataset.

    Args:
        dataset (pandas.DataFrame): The dataset to be used for testing the model.
            The first element of the dataset is the target variable (`y`),
            and everything else is the feature variable (`X`).
        model_selection: The machine learning model to be tested.
        random_state (int): Random seed for reproducibility. Default is 1.
        Prediction (bool): Indicates whether to perform prediction and return prediction results.
            Default is True.

    Returns:
        dict or model: If `Prediction` is True, a dictionary containing the prediction results is returned.
            The keys are 'y_true' (true target values), 'y_pred' (predicted target values),
            and 'accuracy' (accuracy score of the model).
            If `Prediction` is False, the trained model is returned.
    """


    model_list = []
    list_of_histories =[]
    for dataset in data_list:
        y = dataset.iloc[:,[0]]
        X = dataset.drop(columns = dataset.columns[0])



        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(),
                                                        test_size=.3,
                                                        random_state=random_state)


        model = model_select

        model.fit(X_train, y_train)


        MBTI_type = []

        type1 = y.type.value_counts().index.to_list()[0]
        type2 = y.type.value_counts().index.to_list()[1]

        if Prediction == True:
            prediction_dict = PredictDict(model, X_test, y_test)
            print(f"F1-score:{100*round(prediction_dict['macro avg']['f1-score'],5)}")
            print(f"Model Type: {type1 + type2}")

            list_of_histories.append(prediction_dict)


        model_list.append(model)


    return model_list, list_of_histories
# def train_4_types_model(model):



# LOAD/SAVE THE MODELS VIA PICKLE?

def predict_model(model_list,
                  text):
    i = 0
    MBTI_type = []
    for model in model_list:
        prediction = model.predict(text[i])
        print(f"Testing prediction = {prediction}")
        MBTI_type.append(prediction)
        i += 1
    print(f"Final list of predictions = {MBTI_type}")
    return MBTI_type
