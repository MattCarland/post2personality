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
                new_models = False
                ):
    """
    Initializes and saves machine learning models based on the provided parameters or creates new models if specified.

    Args:
        params (list or None): A list of dictionaries containing the parameters for each model. Each dictionary represents the
            parameters for a specific model. The length of the list should be 4. If set to None, new models will be created
            instead of using predefined parameters. (default: None)
        model_type (str): The type of model to initialize. Supported values are 'SGD', 'XGB', and 'ADA'. (default: 'SGD')
        new_models (bool): If True, new models will be created even if parameters are provided. If False, the model_list will
            be built based on the provided parameters. (default: False)

    Returns:
        list: A list containing the initialized machine learning models.

    Raises:
        NotImplementedError: Raised if `params` is None.

    Note:
        - The function saves each initialized model as a pickle file named 'model{i}.pkl', where 'i' represents the index of
          the model in the model_list.
        - The function uses different initialization parameters for each model type:
            - For 'SGD' models, the loss function is set to 'log_loss', the maximum number of iterations is set to 5000,
              and early stopping is enabled if parameters are provided.
            - For 'XGB' models, the loss function is set to 'log_loss' and the number of estimators is set to 500 if
              parameters are provided.
            - For 'ADA' models, the loss function is set to 'log_loss' if parameters are provided.
            - If `model_type` is None, a NameError will be printed, and the function will return NameError.
            - If `params` is None and `new_models` is False, a message will be printed asking to specify parameters, and
              NotImplementedError will be raised.

    """

    ## Building the Model List with Parameters from a grid search
    if params is not None:
        model_list = []
        for i in range(4):
            param = params[i]
            if model_type == 'SGD':
                model = SGDClassifier(loss='log_loss', max_iter=5000, early_stopping=True).set_params(**param)
            if model_type == 'XGB':
                model = xgb(loss = "log_loss", n_estimators=500).set_params(**param)
            if model_type == 'ADA':
                model = ada( n_estimators=100).set_params(**param)
            with open(f'model/model{i}.pkl', 'wb') as file:
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
                    model = xgb(loss = "log_loss", n_estimators=500)
                if model_type == 'ADA':
                    model = ada(n_estimators=100)
                if model_type == None:
                    print("No model specified")
                    return NameError
                with open(f'model/model{i}.pkl', 'wb') as file:
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
        with open(f'model/model{i}.pkl', 'wb') as file:
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
        with open(f'model/model{i}.pkl', 'rb') as file:
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
                    verbose = True,
                    rows_for_search=1000
                    ):
    """
    Performs grid search on multiple datasets using a specified machine learning model type and returns a list of the best
    parameters found for each dataset.

    Args:
        data_list (list): A list of pandas DataFrames representing multiple datasets on which grid search will be performed.
        model_type (str): The type of machine learning model to use. Supported values are 'SGD', 'XGB', and 'ADA'.
            (default: 'SGD')
        verbose (bool): If True, prints detailed information during the grid search process. If False, no additional
            information is printed. (default: True)
        rows_for_search (int): The number of rows to sample from each dataset for the grid search. (default: 1000)

    Returns:
        list: A list of dictionaries containing the best parameters found for each dataset. Each dictionary represents the
            best parameters for a specific dataset.

    Note:
        - The function assumes that each dataset in `data_list` has the target variable in the first column, and the features
          in the remaining columns.
        - The function randomly samples `rows_for_search` rows from each dataset for the grid search.
        - The function splits the sampled data into training and testing sets using a 70:30 ratio.
        - The function loads the machine learning model from a pickle file named 'model/model{i}.pkl', where 'i' represents
          the index of the model to use.
        - The function sets the grid search parameters based on the specified `model_type`:
            - For 'SGD' models, the grid search parameters include 'penalty' (l2, l1, elasticnet) and 'alpha' values.
            - For 'XGB' models, the grid search parameters include 'learning_rate' and 'max_depth' values.
            - For 'ADA' models, the grid search parameters include 'learning_rate' and 'max_depth' values.
        - The grid search uses 5-fold cross-validation.
        - The function prints detailed information about each grid search iteration if `verbose` is True.

    """



    i = 0
    param_list = []


    for dataset in data_list:
        # Set X and y parameters
        y = dataset.iloc[:,[0]]
        X = dataset.drop(columns = dataset.columns[0])



        # Set rows for search
        y = y.sample(n = rows_for_search, ignore_index=True)
        X = X.sample(n = rows_for_search, ignore_index=True)

        # Set X_train and y_train
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(),
                                                test_size=.3)
        # MBTI Types
        type1 = y.type.value_counts().index.to_list()[0]
        type2 = y.type.value_counts().index.to_list()[1]

        # Load Models
        with open(f'model/model{i}.pkl', 'rb') as file:
            model = pickle.load(file)

        # Set parameters to search for
        if model_type == 'SGD':
            param_grid = {'penalty': ['l2', 'l1', 'elasticnet'],
                        'alpha': [0.0001, 0.001, 0.01, .1]}
        if model_type == 'XGB':
            param_grid = {'learning_rate': [.01, .001],
                        #   'n_estimators': [100, 200, 500]
                        'max_depth': [3, 5]
                        }
        if model_type == 'ADA':
            param_grid = {'learning_rate': [.1, .01, .001]
                          # 'n_estimators': [50, 100, 500]
                        #   'max_depth': [3, 5, 8]
                        }

        # Grid Search
        grid_search = GridSearchCV(model,
                            param_grid,
                            cv=5,
                            scoring='accuracy'
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



def train_model(df_list,
            model_list,
            random_state = 1,
            Prediction = True
            ):

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
    for dataset in df_list:
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
        with open(f'model/model{i}.pkl', 'wb') as file:
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
              verbose = True
              ):
    """
    Predicts the MBTI (Myers-Briggs Type Indicator) type for a given list of texts using pre-trained machine learning models.
    Returns a dictionary containing the overall type prediction, a type score, and specific type scores for each individual text.

    Args:
        texts (list): A list of pandas DataFrames representing the texts to predict the MBTI type for. Each DataFrame should
            contain the features required for prediction.
        verbose (bool): If True, prints detailed information for each prediction. If False, no additional information is
            printed. (default: True)

    Returns:
        dict: A dictionary containing the overall type prediction, a type score, and specific type scores for each individual
            text. The dictionary has the following structure:
            {
                "type_prediction": <overall_type_prediction>,
                "type_score": <type_score>,
                "specific_type_score": [
                    {
                        "Type": <individual_type_prediction>,
                        "<class_1>_Probability": <probability_of_class_1>,
                        "<class_2>_Probability": <probability_of_class_2>
                    },
                    ...
                ]
            }

    Note:
        - The function assumes that each text in `texts` is represented as a pandas DataFrame, where each column represents a
          feature required for prediction.
        - The function loads the pre-trained machine learning model from a pickle file named 'model/model{i}.pkl', where 'i'
          represents the index of the model to use.
        - The function makes predictions for each text and calculates the dominance of the predicted class based on the
          predicted probabilities.
        - The function prints detailed information about each prediction if `verbose` is True.
        - The overall type prediction is obtained by concatenating the individual type predictions.
        - The type score is calculated as the average dominance of the predicted class across all predictions.
        - The specific type scores provide the predicted MBTI type, probability of class 1, and probability of class 2 for each
          individual text.

    """

    i = 0
    MBTI_type = []
    Class_Dominance_List = []
    specific_predictions = []

    while i < len(texts):

        with open(f'model/model{i}.pkl', 'rb') as file:
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

    type_score = sum(Class_Dominance_List) / len(Class_Dominance_List)

    result_dict = {"type_prediction": final_type,
                   "type_score": type_score,
                   "specific_type_score": specific_predictions}

    return result_dict
