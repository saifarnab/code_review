import numpy as np
import pandas as pd
from keras import layers
from keras.layers import Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer


class _OperationModel:
    def __init__(self):
        self.df = None
        self.each_set_len = 0
        self.operation_dataframe = []
        self.dense_layers = 512
        self.dropout_layers = 2
        self.dropout = 0.2
        self.activation = ['relu', 'softmax']
        self.loss_function = 'categorical_crossentropy'
        self.optimizer = 'adam'
        self.metrics = ['accuracy']
        self.epochs = 2
        self.batch_size = 64
        self.fold = 10

    def _operation_model(self):
        for i in range(0, self.fold):

            messages = self.df['message'].values
            operations = self.df['operation'].values

            if (len(self.df) - (i + 1) * self.each_set_len) < self.each_set_len:
                train_x = np.delete(messages, list(range(i * self.each_set_len, len(self.df))))
                train_y = np.delete(operations, list(range(i * self.each_set_len, len(self.df))))
                test_x = messages[i * self.each_set_len: len(self.df)]
                test_y = operations[i * self.each_set_len: len(self.df)]
                # print(
                #     f"current range: {i * self.each_set_len, len(self.df)} | train data len {len(train_x)} | test "
                #     f"data len {len(test_x)}")
            else:
                train_x = np.delete(messages, list(range(i * self.each_set_len, (i + 1) * self.each_set_len)))
                train_y = np.delete(operations, list(range(i * self.each_set_len, (i + 1) * self.each_set_len)))
                test_x = messages[i * self.each_set_len: (i + 1) * self.each_set_len]
                test_y = operations[i * self.each_set_len:(i + 1) * self.each_set_len]
                # print(
                #     f"current range: {i * self.each_set_len, (i + 1) * self.each_set_len} | train data len"
                #     f" {len(train_x)} | test data len {len(test_x)}")

            vectorizer = CountVectorizer()
            vectorizer.fit(train_x)
            y_binary = to_categorical(train_y)

            train_vector = vectorizer.transform(train_x)
            input_dim = train_vector.shape[1]  # Number of features
            model = Sequential()
            model.add(layers.Dense(self.dense_layers, input_dim=input_dim, activation=self.activation[0]))
            model.add(Dropout(self.dropout))
            model.add(layers.Dense(self.dense_layers, activation=self.activation[0]))
            model.add(Dropout(self.dropout))
            model.add(layers.Dense(3, activation=self.activation[1]))

            model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
            _ = model.fit(train_vector, y_binary, epochs=self.epochs, batch_size=self.batch_size)

            for idx, item in enumerate(test_x):
                v_temp = vectorizer.transform([item])
                v = model.predict(x=v_temp)
                if test_y[idx] == 0:
                    model_probability = v[0][0]
                else:
                    model_probability = v[0][1]

                self.operation_dataframe.append([item, test_y[idx], model_probability])

    def process(self, df):
        self.df = df
        self.each_set_len = int(len(df) / self.fold)
        self._operation_model()
        return pd.DataFrame(self.operation_dataframe, columns=['message', 'operation', 'prediction'])


operation_model = _OperationModel()
