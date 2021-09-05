from keras import Sequential, optimizers
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Cnn:
    """Model object"""

    def __init__(self, window, location_index, features_index, learning_rate):
        self.window = window
        self.history = None
        self.model = Sequential()
        self.model.add(
            Conv2D(
                padding='same',
                kernel_size=2,
                filters=64,
                activation='relu',
                input_shape=(
                    window,
                    location_index,
                    features_index)))
        self.model.add(MaxPool2D(pool_size=2, padding='same'))
        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(1))
        adam = optimizers.Adam(lr=learning_rate, decay=1e-6)
        self.model.compile(loss='mae', optimizer=adam)

    def fit(self, X, y, batch_size, epochs, verbose):
        self.history = self.model.fit(
            X, y, batch_size=batch_size, epochs=epochs, verbose=verbose)
        return self.history

    def predict(self, X):
        prediction = self.model.predict(X)
        return prediction


class TimeSeries:
    """Prediction Object"""

    def __init__(self, series):
        self.series = series

    def series_division(self, window):
        """data set division"""
        X, y = [], []
        for index in range(len(self.series) - window):
            X.append(self.series[index: index + window])
            y.append(self.series[index + window][-1][0])
        # """The last week sample"""
        # X_train = X[: 69960]
        # X_test = X[-168:]
        # y_train = y[: 69960]
        # y_test = y[-168:]
        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=168, random_state=10,
        #                                                   shuffle=False)
        """random sample"""
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=168, shuffle=False)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=168, shuffle=False)

        return np.array(X_train), np.array(y_train), np.array(
            X_test), np.array(y_test), np.array(X_val), np.array(y_val)
        # return np.array(X), np.array(y)

    def predict_temperatures(
            self,
            model,
            window,
            sample_size=168,
            feature_temperature=2):
        """predict recursively"""
        predictions = np.zeros((sample_size, 1))
        for i in range(len(self.series) - sample_size, len(self.series)):
            prediction = model.predict(
                X=self.series[np.newaxis, i - window: i])[0, 0]
            predictions[i - len(self.series) + sample_size] = prediction
            self.series[i, -1, feature_temperature] = prediction
        return predictions
