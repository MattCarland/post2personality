# Can you make RNN from pytorch?

# Or does it HAVE to be TensorFlow?

import numpy as np
import pickle
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import cross_val_predict, train_test_split

def initialize_models(metrics,
                      loss
                      ):
    model_list = []
    for i in range(4):
        model = Sequential([
        layers.Embedding(input_dim=vocab_size+1, input_length=maxlen, output_dim=embedding_size, mask_zero=True),
        layers.Conv1D(10, kernel_size=15, padding='same', activation="relu"),
        layers.Conv1D(10, kernel_size=10, padding='same', activation="relu"),
        layers.Flatten(),
        layers.Dense(30, activation='relu'),
        layers.Dropout(0.15),
        layers.Dense(1, activation='relu'),
    ])
        model.compile(loss = loss, optimizer=Adam, metrics = metrics)
        with open(f'model/modelDL{i}.pkl', 'wb') as file:
            pickle.dump(model, file)
        model_list.append(model)
    return model_list



def train_models(df_list,
                 model_list):
    list_of_histories = []
    i = 0
    for dataset in df_list:
        # Set X and y
        y = dataset.iloc[:,[0]]
        X = dataset.drop(columns = dataset.columns[0])
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=.3)

        # MBTI Types per DF
        type1 = y.type.value_counts().index.to_list()[0]
        type2 = y.type.value_counts().index.to_list()[1]

        # Fit Model and Save the Fitted Model
        model = model_list[i]
        model.fit(X_train, y_train)
        with open(f'model/modelDL{i}.pkl', 'wb') as file:
            pickle.dump(model, file)
        i += 1
