# Libraries
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
import pickle

from sklearn.ensemble import GradientBoostingClassifier as xgb
from sklearn.ensemble import AdaBoostClassifier as ada

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def initialize_models(params = None,
                      model_type = 'SGD',
                      new_models = False):
    """
    Initialize and save machine learning models.

    Parameters:
        params (list): A list of parameter dictionaries for model initialization and configuration.
                       Each dictionary contains the model-specific parameters.
        is_SGD (bool): If True, the models are initialized using SGDClassifier. Default is True.
        new_models (bool): If True, new models are created from scratch using SGDClassifier.
                           Default is False.

    Returns:
        list or str: A list of initialized models if params or new_models is True, or a string
                     requesting the specification of parameters in list format.

    """

    ## Building the Model List with Parameters from a grid search
    if params is not None:
        model_list = []
        for i in range(4):
            param = params[i]
            if model_type == 'SGD':
                model = SGDClassifier(loss='log_loss', max_iter=5000, early_stopping=True).set_params(**param)
            if model_type == 'XGB':
                model = xgb(loss = "log_loss").set_params(**param)
            if model_type == 'ADA':
                model = ada(loss = "log_loss").set_params(**param)
            with open(f'model{i}.pkl', 'wb') as file:
                pickle.dump(model, file)
            model_list.append(model)
        return model_list

    ## Building the Model List with New Models
    if new_models == True:
            model_list = []
            for i in range(4):
                if model_type == 'SGD':
                    model = SGDClassifier(loss='log_loss', max_iter=5000, early_stopping=True)
                if model_type == 'XGB':
                    model = xgb(loss = "log_loss", n_estimators=1000)
                if model_type == 'ADA':
                    model = ada(loss = "log_loss")
                if model_type == None:
                    print("No model specified")
                    return NameError
                with open(f'model{i}.pkl', 'wb') as file:
                    pickle.dump(model, file)
                model_list.append(model)
            return model_list

    if params is None:
        print("Please specify parameters in list format")
        return NotImplementedError

    return model_list


def save_models_pkl(model_list):

    """
    Save a list of machine learning models as pickle files.

    Parameters:
        model_list (list): A list of machine learning models to be saved.

    Returns:
        None

    Side Effects:
        - Saves each model in model_list as a pickle file named 'model{i}.pkl',
          where i represents the index of the model in the list.
        - Prints the model name and status message for each saved model.

    """

    i = 0
    for model in model_list:
        with open(f'model{i}.pkl', 'wb') as file:
            pickle.dump(model, file)
        print(model, " stored")
        i += 1


def load_models_pkl():

    """
    Load machine learning models from pickle files.

    Returns:
        list: A list of loaded machine learning models.

    Side Effects:
        - Loads each model from the pickle files named 'model{i}.pkl', where i represents the index.
        - Prints the loaded model list.

    """

    i = 0
    model_list = []
    while i < 4:
        with open(f'model{i}.pkl', 'rb') as file:
            model = pickle.load(file)
        model_list.append(model)
    print(f"Model list loaded:")
    print(model_list)
    return model_list




def PredictDict(model, X_test, y_test):
    """
    Generate a dictionary containing classification report metrics and balanced accuracy score.

    Parameters:
        model: The trained machine learning model used for prediction.
        X_test: The input features used for prediction.
        y_test: The ground truth labels for the test data.

    Returns:
        dict: A dictionary containing the classification report metrics, including precision, recall,
              F1-score, support, and the balanced accuracy score.

    """
    y_pred = cross_val_predict(model, X_test, y_test, cv = 10)
    report = classification_report(y_test, y_pred, output_dict=True)
    bal_acc = balanced_accuracy_score(y_pred, y_test)
    report['bal_acc'] = bal_acc
    return report



@ignore_warnings(category=ConvergenceWarning)
def grid_search_all_models(data_list,
                           model_type='SGD',
                           verbose = True):

    """
    Perform grid search for hyperparameter tuning on multiple models using different datasets.

    Parameters:
        data_list (list): A list of datasets to be used for grid search.
        is_SGD (bool): If True, the models to be tuned are based on SGDClassifier. Default is True.
        verbose (bool): If True, print information about the grid search process. Default is True.

    Returns:
        list: A list of dictionaries, where each dictionary contains the best parameters
              found during the grid search for each model.

    Side Effects:
        - Prints information about the grid search process, including the model being searched,
          the best parameters found, and the best score achieved, if verbose is True.

    """


    i = 0
    param_list = []


    for dataset in data_list:
        # Set X and y parameters
        y = dataset.iloc[:,[0]]
        X = dataset.drop(columns = dataset.columns[0])

        # Set X_train and y_train
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(),
                                                test_size=.3)
        # MBTI Types
        type1 = y.type.value_counts().index.to_list()[0]
        type2 = y.type.value_counts().index.to_list()[1]

        # Load Models
        with open(f'model{i}.pkl', 'rb') as file:
            model = pickle.load(file)

        # Set parameters to search for
        if model_type == 'SGD':
            param_grid = {'penalty': ['l2', 'l1', 'elasticnet'],
                        'alpha': [0.0001, 0.001, 0.01, .1]}
        if model_type == 'XGB':
            param_grid = {'learning_rate': [.1, .01, .001]
                        #   'n_estimators': [100, 200, 500]
                        #   'max_depth': [3, 5, 8]
                        }
        if model_type == 'ADA':
            param_grid = {'learning_rate': [.1, .01, .001],
                          'n_estimators': [50, 100, 500]
                        #   'max_depth': [3, 5, 8]
                        }

        # Grid Search
        grid_search = GridSearchCV(model,
                                   param_grid,
                                   cv=5
                                   )
        grid_search.fit(X_train, y_train)

        param_list.append(grid_search.best_params_)

        if verbose:
            print(f"Model Searched = {model} {(type1+type2).upper()}")
            print("================================")
            print("Best parameters:", grid_search.best_params_)
            print("Best score:", grid_search.best_score_)
            print("================================")

    return param_list



def train_model(data_list,
                 random_state = 1,
                 Prediction = True):

    """
    Train machine learning models on multiple datasets.

    Parameters:
        data_list (list): A list of datasets to be used for training.
        model_list (list): A list of machine learning models to be trained.
        random_state (int): Random seed for reproducibility. Default is 1.
        Prediction (bool): If True, generate predictions and print evaluation metrics. Default is True.

    Returns:
        list: A list of dictionaries containing evaluation metrics for each trained model.

    Side Effects:
        - Saves each trained model as a pickle file named 'model{i}.pkl', where i represents the index.
        - Prints evaluation metrics if Prediction is True, including the F1-score, model information,
          and the corresponding MBTI type.
        - Appends prediction dictionaries to the list_of_histories.

    """


    list_of_histories = []
    i = 0
    for dataset in data_list:
        # Set X and y
        y = dataset.iloc[:,[0]]
        X = dataset.drop(columns = dataset.columns[0])
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(),
                                                        test_size=.3,
                                                        random_state=random_state)

        # MBTI Types per DF
        type1 = y.type.value_counts().index.to_list()[0]
        type2 = y.type.value_counts().index.to_list()[1]

        # Fit Model and Save the Fitted Model
        model = model_list[i]
        model.fit(X_train, y_train)
        with open(f'model{i}.pkl', 'wb') as file:
            pickle.dump(model, file)

        # Save the results and print them
        if Prediction == True:
            prediction_dict = PredictDict(model, X_test, y_test)
            print(f"F1-score: {100*round(prediction_dict['macro avg']['f1-score'],5)}%")
            print(f"Model: {(model)}")
            print(f"MBTI Type: {(type1 + type2).upper()}")
            print("========================================")

            list_of_histories.append(prediction_dict)

        i += 1

    return list_of_histories


def predict_model(texts,
                  verbose = True):
    """
    Predict the MBTI type based on given texts using pre-trained models.

    Parameters:
        texts (list): A list of text inputs to be used for prediction.
        verbose (bool): If True, print detailed prediction information for each input. Default is True.

    Returns:
        str: The predicted MBTI type based on the input texts.

    Side Effects:
        - Loads pre-trained models from pickle files named 'model{i}.pkl', where i represents the index.
        - Prints the testing prediction, probability of each class, and the final prediction if verbose is True.
        - Appends each individual prediction to the MBTI_type list.

    """

    i = 0
    MBTI_type = []
    Class_Dominance_List = []
    specific_predictions = []

    while i < len(texts):

        with open(f'model{i}.pkl', 'rb') as file:
            model = pickle.load(file)

        df = texts[i]
        for col in df.columns.values:
            if col == 'type':
                df.drop('type', axis=1, inplace=True)

        prediction = model.predict(df)[0].upper()
        proba_raw = model.predict_proba(df)[0]
        class_1 = model.classes_[0].upper()
        class_2 = model.classes_[1].upper()


        if proba_raw[0] > proba_raw[1]:
            Class_Dominance_List.append(proba_raw[0])
        else:
            Class_Dominance_List.append(proba_raw[1])

        if verbose:
            print(f"Testing prediction = {prediction}")
            print(f"Probability of {class_1}: {100*(proba_raw[0].round(5))}%")
            print(f"Probability of {class_2}: {100*(proba_raw[1].round(5))}%")
            print("\n")

        MBTI_type.append(prediction)

        type_dict = {"Type" : prediction,
                 f"{class_1} Probability": proba_raw[0],
                 f"{class_2} Probability": proba_raw[1]}

        specific_predictions.append(type_dict)

        i += 1

    final_type = ''.join(MBTI_type).upper()

    # print(Class_Dominance_List)
    type_score = sum(Class_Dominance_List) / len(Class_Dominance_List)

    result_dict = {"type_prediction": final_type,
                   "type_score": type_score,
                   "specific_type_score": specific_predictions}

    return result_dict
