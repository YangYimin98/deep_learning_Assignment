from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


class Cnn:
    def __init__(self, input_size):
        self.input_size = input_size
        self.history = None
        self.model = Sequential()
        self.model.add(
            Conv1D(
                filters=64,
                kernel_size=2,
                activation='relu',
                batch_input_shape=(
                    None,
                    self.input_size,
                    1)))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs, verbose=0, batch_size=32):
        self.history = self.model.fit(
            X, y, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def predict(self, x):
        x_input = x.reshape((1, self.input_size, 1))
        prediction = self.model.predict(x_input, verbose=0)
        return prediction[0]
