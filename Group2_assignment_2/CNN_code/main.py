from utils import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import scipy.io as io


def series_division(series):
    X, y = [], []
    for index in range(len(series) - window):
        X.append(series[index: index + window])
        y.append(series[index + window][-1][0])
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=168, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=168, shuffle=False)
    return np.array(X_train), np.array(y_train), np.array(
        X_test), np.array(y_test), np.array(X_val), np.array(y_val)


def predict_temperatures(
        model,
        window,
        sample_size=168,
        feature_temperature=2):
    """predict recursively"""
    temp = series.copy()
    predictions = np.zeros((sample_size, 1))
    for i in range(len(temp) - sample_size, len(temp)):
        prediction = model.predict(X=temp[np.newaxis, i - window: i])[0, 0]
        predictions[i - len(temp) + sample_size] = prediction
        temp[i, -1, feature_temperature] = prediction
    return predictions


if __name__ == '__main__':
    """load data"""
    series_test = io.loadmat('data.mat')['X'].astype(float)
    series = series_test.copy()
    # extract the last feature and train all the cities, then extract one city
    # in the following
    series = series[:, :, [2]]
    series_use = series.copy()
    """hyper-parameters tuning"""
    # window = [20, 30, 40, 50, 60, 70, 80, 90]  # After experiments, the best
    # one is 80
    window = 80
    # learning_rate = [0.1, 0.01, 0.001, 0.0001]  # After experiments, the
    # best one is 0.01
    learning_rate = 0.01
    # batch_size = [32, 64, 128]  # After experiments, the best one is 32
    batch_size = 50
    # optimizer_type = ['adam', 'adagrad', 'rmsprop', 'sgd']  # After
    # experiments, the best one is adam
    epoch = 3 

    """Normalization data set"""
    scales = []
    for feature in range(series.shape[2]):
        scale = MinMaxScaler(feature_range=(0, 1))
        scale.fit(series[..., feature])
        series[..., feature] = scale.transform(series[..., feature])
        scales.append(scale)
    # print(scales)

    X_train, y_train, X_test, y_test, X_val, y_val = series_division(series)
    # print('X_train.shape:{0}'.format(X_train.shape))
    # print('X_test.shape:{0}'.format(X_test.shape))
    # print('X_val.shape:{0}'.format(X_val.shape))
    location_index = series.shape[1]
    feature_index = series.shape[2]
    # print(location_index)
    # print(feature_index)
    model = Cnn(window, location_index, feature_index, learning_rate)
    history_train = model.fit(
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epoch,
        verbose=1)
    # history_val = model.fit(X=X_val, y=y_val, batch_size=batch_size, epochs=epoch, verbose=1)
    # loss_train = history_train.history['loss']
    # loss_val = history_val.history['loss']
    # print(loss_train)

    """Experiments for window"""
    # for w in window:
    #     X_train, y_train, X_test, y_test, X_val, y_val = TS.series_division(w)
    #     model = Cnn(w, location_index, feature_index, learning_rate)
    #     history_train = model.fit(X=X_train, y=y_train, batch_size=batch_size, epochs=epoch, verbose=1)
    #     # history_val = model.fit(X=X_val, y=y_val, batch_size=batch_size, epochs=epoch, verbose=1)
    #
    #     """plot the MSE loss function"""
    #     loss_train = history_train.history['loss']
    #     print(loss_train)
    #
    #     # loss_val = history_val.history['loss']
    #     epochs = range(1, len(loss_train) + 1)
    #     # plt.plot(epochs, loss_val, label='windows' + str(w))
    #     plt.plot(epochs, loss_train, label='windows: ' + str(w))
    #     # plt.plot(epochs, loss_train, 'green', label='train set loss')
    #     plt.grid()
    #     plt.xlabel('Epochs')
    #     plt.ylabel('MSE Loss')
    #     plt.title('MSE Loss VS Epochs')
    # plt.legend()
    # plt.show()
    #
    """Experiments for learning rate"""
    # for l_r in learning_rate:
    #     X_train, y_train, X_test, y_test, X_val, y_val = TS.series_division(window)
    #     model = Cnn(window, location_index, feature_index, l_r)
    #     history_train = model.fit(X=X_train, y=y_train, batch_size=batch_size, epochs=epoch, verbose=0)
    #     # history_val = model.fit(X=X_val, y=y_val, batch_size=batch_size, epochs=epoch, verbose=1)
    #     """plot the MSE loss function"""
    #     loss_train = history_train.history['loss']
    #     print(loss_train)
    #     # loss_val = history_val.history['loss']
    #     epochs = range(1, len(loss_train) + 1)
    #     # plt.plot(epochs, loss_val, label='windows' + str(w))
    #     plt.plot(epochs, loss_train, label='l_r: ' + str(l_r))
    #     # plt.plot(epochs, loss_train, 'green', label='train set loss')
    #     plt.grid()
    #     plt.xlabel('Epochs')
    #     plt.ylabel('MSE Loss')
    #     plt.title('MSE Loss VS Epochs')
    # plt.legend()
    # plt.show()
    #
    """Experiments for batch size"""
    # for batch in batch_size:
    #     X_train, y_train, X_test, y_test, X_val, y_val = TS.series_division(window)
    #     model = Cnn(window, location_index, feature_index, learning_rate)
    #     history_train = model.fit(X=X_train, y=y_train, batch_size=batch, epochs=epoch, verbose=1)
    #     # history_val = model.fit(X=X_val, y=y_val, batch_size=batch_size, epochs=epoch, verbose=1)
    #     """plot the MSE loss function"""
    #     loss_train = history_train.history['loss']
    #     # loss_val = history_val.history['loss']
    #     epochs = range(1, len(loss_train) + 1)
    #     # plt.plot(epochs, loss_val, label='windows' + str(w))
    #     plt.plot(epochs, loss_train, label='batch size ' + str(batch))
    #     # plt.plot(epochs, loss_train, 'green', label='train set loss')
    #     plt.grid()
    #     plt.xlabel('Epochs')
    #     plt.ylabel('MSE Loss')
    #     plt.title('MSE Loss VS Epochs')
    # plt.legend()
    # plt.show()

    """Experiments for optimizers"""

    # part1: sgd
    # model = Cnn(window, location_index, feature_index, learning_rate)
    # history_train = model.fit(X=X_train, y=y_train, batch_size=batch_size, epochs=epoch, verbose=1)
    # # history_val = model.fit(X=X_val, y=y_val, batch_size=batch_size, epochs=epoch, verbose=1)
    # """plot the MSE loss function"""
    # loss_train = history_train.history['loss']
    # print(loss_train)
    # [0.016394061228465172, 0.008851141818750557, 0.007075452466472036, 0.00603864140821835,
    # 0.005393629689579704, 0.004976970383554136, 0.004694177552485667,
    # 0.004492683762362552, 0.004335391007879797, 0.00421147075529166]

    # part2: adam
    # model = Cnn(window, location_index, feature_index, learning_rate)
    # history_train = model.fit(X=X_train, y=y_train, batch_size=batch_size, epochs=epoch, verbose=1)
    # # history_val = model.fit(X=X_val, y=y_val, batch_size=batch_size, epochs=epoch, verbose=1)
    # """plot the MSE loss function"""
    # loss_train = history_train.history['loss']
    # print(loss_train)
    # [0.4591913114719534, 0.004886740994645466, 0.003899662330648752, 0.0036269508972114205, 0.0035033992785626555,
    # 0.003520686471119133, 0.0034547613220869047, 0.0034603069612056544,
    # 0.003439152570585007, 0.0034707981823980535]

    # # part3: rmsprop
    # model = Cnn(window, location_index, feature_index, learning_rate)
    # history_train = model.fit(X=X_train, y=y_train, batch_size=batch_size, epochs=epoch, verbose=1)
    # # history_val = model.fit(X=X_val, y=y_val, batch_size=batch_size, epochs=epoch, verbose=1)
    # """plot the MSE loss function"""
    # loss_train = history_train.history['loss']
    # print(loss_train)
    # [18.45614579155428, 0.024815223464904328, 0.018660769764302327, 0.00858051223631855, 0.008010468289411281,
    # 0.005797357702266963, 0.004826672880007016, 0.0046458499869656625,
    # 0.004482391280507285, 0.004396934286710058]

    # # part4: adagrad
    # model = Cnn(window, location_index, feature_index, learning_rate)
    # history_train = model.fit(X=X_train, y=y_train, batch_size=batch_size, epochs=epoch, verbose=1)
    # # history_val = model.fit(X=X_val, y=y_val, batch_size=batch_size, epochs=epoch, verbose=1)
    # """plot the MSE loss function"""
    # loss_train = history_train.history['loss']
    # print(loss_train)
    # [0.6719776731426277, 0.007375514103984214, 0.005651775034243274, 0.004939298278106847, 0.004542891737069385,
    # 0.004351650752819304, 0.004180787360574174, 0.004129592630575504,
    # 0.003998723563084937, 0.003952638077601364]

    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # y1 = [0.016394061228465172, 0.008851141818750557, 0.007075452466472036, 0.00603864140821835,
    # 0.005393629689579704, 0.004976970383554136, 0.004694177552485667,
    # 0.004492683762362552, 0.004335391007879797, 0.00421147075529166]
    # y2 = [0.4591913114719534, 0.004886740994645466, 0.003899662330648752, 0.0036269508972114205, 0.0035033992785626555,
    # 0.003520686471119133, 0.0034547613220869047, 0.0034603069612056544, 0.003439152570585007, 0.0034707981823980535]
    # y3 = [18.45614579155428, 0.024815223464904328, 0.018660769764302327, 0.00858051223631855, 0.008010468289411281,
    # 0.005797357702266963, 0.004826672880007016, 0.0046458499869656625, 0.004482391280507285, 0.004396934286710058]
    # y4 = [0.6719776731426277, 0.007375514103984214, 0.005651775034243274, 0.004939298278106847, 0.004542891737069385,
    # 0.004351650752819304, 0.004180787360574174, 0.004129592630575504, 0.003998723563084937, 0.003952638077601364]
    # plt.plot(x, y1, label='sgd')
    # plt.plot(x, y2, label='adam')
    # plt.plot(x, y3, label='rmsprop')
    # plt.plot(x, y4, label='adagrad')
    # plt.ylim(0, 0.1)
    # plt.legend()
    # plt.grid()
    # plt.xlabel('Epochs')
    # plt.ylabel('MSE loss')
    # plt.title('MSE Loss VS Epochs')
    # plt.show()

    y_predict_recursively = predict_temperatures(
        model, window, sample_size=168, feature_temperature=0)  # predict recursively
    # """calculate the MAE"""
    # mae_recursively = mean_absolute_error(y_test, y_predict_recursively)
    # print('The mae result is {0}: '.format(mae_recursively))

    """plot the prediction images"""
    # plt.figure(figsize=(8, 4))
    # plt.plot(y_predict_recursively[:, 0], label="Prediction")
    # plt.plot(series[-168:, 3, 0], label="Real Data")
    # plt.legend()
    # plt.grid()
    # plt.xlabel("Time Index (hour)")
    # plt.ylabel("Temperature(C)")
    # plt.title('Real data and prediction')
    # plt.show()

    """scale back to compare"""

    series_orig_rec = np.zeros((len(y_predict_recursively), series.shape[1]))
    series_orig_rec[:, 2:3] = y_predict_recursively
    y_test_orig = scales[0].inverse_transform(series[-168:, :, 0])
    y_pred_rec_orig = scales[0].inverse_transform(series_orig_rec)
    print(y_test_orig[:, 2] * 0.1)  # times 0.1 cause temperature unit is 0.1C
    print(y_pred_rec_orig[:, 2] * 0.1)
    mae_recursively = mean_absolute_error(y_test_orig, y_pred_rec_orig)
    print('The mae result is {0}: '.format(mae_recursively))

    """plot comparison"""

    plt.figure(figsize=(8, 4))
    plt.plot(y_pred_rec_orig[:, 2] * 0.1, label="Prediction")
    plt.plot(y_test_orig[:, 2] * 0.1, label="Real data")
    plt.grid()
    plt.legend()
    plt.ylim(-5, 10)
    plt.xlabel("Time Index(hour)")
    plt.ylabel("Temperature(C)")
    plt.title('Real and Prediction dta')
    plt.show()
