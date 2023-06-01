# Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
import pickle

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings



def initialize_models(params = None,
                      is_SGD = True,
                      new_models = True):
    model_list = []
    if params is not None:
        for i in range(4):
            param = params[i]
            print(param)
            # param_list = []
            # for k,v in param.items():
            #     param_indiv = f"{k}={v} ,"
            #     param_list.append(param_indiv)
            # param = "".join(param_list)[:-1]
            # print(param)

            if is_SGD == True:
                model = SGDClassifier(max_iter=2000, early_stopping=True).set_params(**param)
            with open(f'model{i}.pkl', 'wb') as file:
                pickle.dump(model, file)
            model_list.append(model)
        return model_list

    if new_models == True:
            model_list = []
            for i in range(4):
                model = SGDClassifier(max_iter=2000, early_stopping=True)
                with open(f'model{i}.pkl', 'wb') as file:
                    pickle.dump(model, file)
            return model_list

    if params is None:
        return "Please specify parameters in list format"

    return model_list




def PredictDict(model, X_test, y_test):
    y_pred = cross_val_predict(model, X_test, y_test, cv = 10)
    report = classification_report(y_test, y_pred, output_dict=True)
    bal_acc = balanced_accuracy_score(y_pred, y_test)
    report['bal_acc'] = bal_acc
    return report

@ignore_warnings(category=ConvergenceWarning)
def grid_search_all_models(data_list,
                           is_SGD=True):
    i = 0
    param_list = []


    for dataset in data_list:
        y = dataset.iloc[:,[0]]
        X = dataset.drop(columns = dataset.columns[0])

        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(),
                                                test_size=.3)

        type1 = y.type.value_counts().index.to_list()[0]
        type2 = y.type.value_counts().index.to_list()[1]

        with open(f'model{i}.pkl', 'rb') as file:
            model = pickle.load(file)

        if is_SGD == True:
            param_grid = {'loss': ['hinge', 'log_loss',
                                     'perceptron',
                                    'squared_error'],
                        'penalty': ['l2', 'l1', 'elasticnet'],
                        'alpha': [0.0001, 0.001, 0.01, .1]}

        grid_search = GridSearchCV(model,
                                   param_grid,
                                   cv=10, n_jobs=-1
                                   )
        grid_search.fit(X_train, y_train)

        param_list.append(grid_search.best_params_)

        print(f"Model Searched = {model} {(type1+type2).upper()}")
        print("================================")
        print("Best parameters:", grid_search.best_params_)
        print("Best score:", grid_search.best_score_)
        print("================================")

    return param_list






def train_model(data_list,
                 model_list,
                 random_state = 1,
                 Prediction = True):


    list_of_histories =[]
    i = 0
    for dataset in data_list:
        y = dataset.iloc[:,[0]]
        X = dataset.drop(columns = dataset.columns[0])



        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(),
                                                        test_size=.3,
                                                        random_state=random_state)


        type1 = y.type.value_counts().index.to_list()[0]
        type2 = y.type.value_counts().index.to_list()[1]

        model = model_list[i]

        model.fit(X_train, y_train)

        with open(f'model{i}.pkl', 'wb') as file:
            pickle.dump(model, file)


        if Prediction == True:
            prediction_dict = PredictDict(model, X_test, y_test)
            print(f"F1-score: {100*round(prediction_dict['macro avg']['f1-score'],5)}%")
            print(f"Model: {(model)}")
            print(f"MBTI Type: {(type1 + type2).upper()}")
            print("========================================")

            list_of_histories.append(prediction_dict)



        i += 1




    return list_of_histories
# def train_4_types_model(model):



# LOAD/SAVE THE MODELS VIA PICKLE?

def predict_model(texts):

    i = 0
    MBTI_type = []

    while i < len(texts):

        with open(f'model{i}.pkl', 'rb') as file:
            model = pickle.load(file)

        df = texts[i]
        for col in df.columns.values:
            if col == 'type':
                df.drop('type', axis=1, inplace=True)

        prediction = model.predict(df)[0]
        proba_raw = model.predict_proba(df)[0]
        proba_max = max(proba_raw)
        proba_min = min(proba_raw)


        print(f"Testing prediction = {prediction}")
        print(f"Probability = {100*(proba_max.round(5))}%")
        print(f"classes = {model.classes_}")
        print("\n")
        MBTI_type.append(prediction)


        i += 1

    final_type = ''.join(MBTI_type).upper()
    print(f"Final Prediction = {final_type}")
    return final_type
