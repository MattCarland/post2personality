# Can you make RNN from pytorch?

# Or does it HAVE to be TensorFlow?

import numpy as np
import pickle
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import cross_val_predict, train_test_split

def initialize_models(metrics,
                      loss,
                      vocab_size,
                      max_len = 768,
                      embedding_size = 768
                      ):
    model_list = []
    for i in range(4):
        model = Sequential([
        layers.Embedding(input_dim=vocab_size+1, input_length=max_len, output_dim=embedding_size, mask_zero=True),
        layers.Conv1D(10, kernel_size=20, padding='same', activation="relu"),
        layers.Conv1D(10, kernel_size=10, padding='same', activation="relu"),
        layers.Flatten(),
        layers.Dense(20, activation='relu'),
        layers.Dropout(0.15),
        layers.Dense(1, activation='relu'),
    ])
        model.compile(loss = loss, optimizer='Adam', metrics = metrics)
        with open(f'model/modelDL{i}.pkl', 'wb') as file:
            pickle.dump(model, file)
        model_list.append(model)
    return model_list



def train_models(df_list,
                 model_list,
                 epochs = 20,
                 batch_size = 32,
                 early_stopping_patience = 2
                 ):
    list_of_histories = []
    i = 0
    es = EarlyStopping(patience = early_stopping_patience)
    for dataset in df_list:
        # Set X and y
        y = dataset.iloc[:,[0]]
        X_pad = dataset['embeddings'].to_list()

        # X_pad = pad_sequences(X, dtype=float, padding = 'post')

        # X_pad = np.asarray(X_pad).astype(np.float32)

        # X_pad = X_pad.reshape

        print(X_pad.shape)

        # MBTI Types per DF
        type1 = y.type.value_counts().index.to_list()[0]
        type2 = y.type.value_counts().index.to_list()[1]

        # Fit Model and Save the Fitted Model
        model = model_list[i]
        model.fit(X_pad, y,
                  validation_split = .25,
                  epochs = epochs,
                  batch_size = batch_size,
                  callbacks = [es])
        with open(f'model/modelDL{i}.pkl', 'wb') as file:
            pickle.dump(model, file)
        i += 1
