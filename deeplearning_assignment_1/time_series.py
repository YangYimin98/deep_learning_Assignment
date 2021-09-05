import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from cnn_model import Cnn
import os
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TimeSeries:
    """This object is used to build the time series model"""

    def __init__(self, series):
        self.series = series

    def series_division(self, windows):
        """Preprocessing the data set by using window and batch to divide"""

        X, y = [], []
        for index in range(len(self.series) - windows - 1):
            X.append(self.series[index: index + windows])
            y.append(self.series[index + windows])
        # print(len(X))
        splitting_num = int(len(X) * 0.8)
        # print(splitting_num)
        x_train = X[: splitting_num]
        x_val = X[splitting_num:]
        y_train = y[: splitting_num]
        y_val = y[splitting_num:]
        # print('training set size is ' + str(len(x_train)))
        # print('training set size is ' + str(len(y_val)))
        return np.array(x_train), np.array(
            y_train), np.array(x_val), np.array(y_val)

    def predict_ahead_data_point(
            self,
            model_used,
            windows,
            previous_point,
            next_points):
        """Predict one step ahead data point"""

        # if the series length is enough, then using the remain part
        if len(self.series) > previous_point:
            prediction = np.zeros(self.series.shape)
            prediction[:previous_point - 1] = self.series[:previous_point - 1]
        else:
            prediction = np.zeros((len(self.series) + next_points, 1))
            prediction[:len(self.series)] = self.series

        for i in range(previous_point, previous_point + next_points):
            x = prediction[i - windows:i]

            prediction[i] = model_used.predict(x)

        return prediction


def load_data(dataset_name):
    """load the data set"""

    data_loading = io.loadmat(dataset_name)
    data = np.array(data_loading['Xtrain'])
    return data


def plot(model_generated, predictions, titles, nums):
    """plot all the visualization images"""

    figure = plt.figure(1)
    for index in range(len(model_generated)):
        figure.add_subplot(nums, 1, index + 1)
        print(index)
        plt.plot(
            predictions[index],
            linewidth=0.5,
            linestyle="-",
            color='black')
        plt.title(titles[index], fontsize=10)
    figure.subplots_adjust(hspace=0.5)
    plt.show()


def train(windows, batch_sizes, epochs, previous_points, next_points):
    """Train to finish different kinds of experiments"""
    models, predictions, titles = [], [], []
    model_in_processing = 1
    model_generated_nums = len(windows) * len(batch_sizes)
    for window in windows:
        x_train, y_train, x_val, y_val = TS_norm.series_division(window)
        for batch_size in batch_sizes:
            title = "Windows:{0}; Batches:{1}; Epochs:{2}".format(
                window, batch_size, epochs)
            print("Model {0}/{1}: {2}".format(model_in_processing,
                                              model_generated_nums, title))
            model = Cnn(window)
            model.fit(
                x_train,
                y_train,
                epochs=epochs,
                verbose=0,
                batch_size=batch_size)
            model.fit(
                x_val,
                y_val,
                epochs=epochs,
                verbose=1,
                batch_size=batch_size)
            prediction = TS_norm.predict_ahead_data_point(
                model_used=model,
                windows=window,
                previous_point=previous_points,
                next_points=next_points
            )
            models.append(model)
            predictions.append(prediction)
            titles.append(title)
            model_in_processing += 1
    if len(windows) > len(batch_sizes):
        plot(models, predictions, titles, len(windows))
    else:
        plot(models, predictions, titles, len(batch_sizes))
    return predictions


series = load_data('Xtrain.mat')
series_test = np.array(io.loadmat('Xtest-1.mat')['Xtest'])
scale = MinMaxScaler(feature_range=(0, 1))
scale.fit(series)
scale.fit(series_test)
TS_norm = TimeSeries(series)
window = 50
