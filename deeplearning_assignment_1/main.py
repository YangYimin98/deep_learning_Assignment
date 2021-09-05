"""Important: Just for expansion, not the highest priority, LSTM is the HIGHEST important one!"""
from time_series import *
from cnn_model import Cnn
from sklearn.metrics import mean_absolute_error
"""Experiments part"""
"""Test different parameters"""

PREVIOUS_POINTS = 1000  # We test 300, 600, 1000
NEXT_POINTS = 200  # Assignment requested value:200
WINDOWS = [50]  # We test 10, 40, 60, 100
BATCH_SIZES = [64]  # We test 32, 64, 128
EPOCHS = 100  # We test 10, 100, 1000

# """Experiment_1: Find the impact of epochs parameters"""
# """Experiment_2: After finding a good epoch number, use it to train windows and batch_size"""
# """Experiment_3: Use the previous parameters to find how many previous time steps can obtain best answer"""
# """Experiment_4: Compare normalization with real measurements"""
# """Experiment_5: Use trained model to predict the next 200 points in a recursive fashion. """
# """Calculate MAE"""

train(windows=WINDOWS, batch_sizes=BATCH_SIZES, epochs=EPOCHS,
      previous_points=PREVIOUS_POINTS, next_points=NEXT_POINTS)

predictions = TS_norm.predict_ahead_data_point(
    model_used=Cnn(window),
    windows=window,
    previous_point=1000,
    next_points=200
)
print(mean_absolute_error(series_test, predictions[-200:])
      )
plt.figure(figsize=(8, 2))
plt.xlabel('Epochs')
plt.ylabel('Amplitude')
# plt.plot(predictions, linewidth=1, linestyle="-", color='blue', label='Real Data')
plt.plot(predictions[:1000], linewidth=1, linestyle="-", color='blue', label='Real Data')
plt.plot(range(1000, 1200), predictions[1000:1200], linewidth=1, linestyle="-", color='black', label='Real Data Predictions')

plt.plot(
    range(
        1000,
        1200),
    series_test,
    linewidth=0.5,
    linestyle="-",
    color='orange',
    label='Prediction')
plt.legend(ncol=3, labels=['Real Data', 'Real Data Predictions', 'Prediction'])
plt.show()


# # """Using MinMaxScaler to finish Normalization, we first do experiments on different parameters to get the best one,
# # then using these parameters to train normalization， then transform back
# # to compare"""
# series = load_data('Xtrain.mat')
# scale = MinMaxScaler(feature_range=(0, 1))
# series_norm = scale.fit(series)
# series_orig = scale.inverse_transform(series_norm)
# TS_norm = TimeSeries(series_norm)
# TS_orig = TimeSeries(series_orig)
#
# window_size = 40
# epochs = 300
#
# X, y = TS_orig.series_division(window_size)
# X_norm, y_norm = TS_norm.series_division(window_size)
#
# cnn_orig = Cnn(window_size)
# cnn_orig.fit(X, y, epochs, 2)
# cnn_norm = Cnn(window_size)
# cnn_norm.fit(X_norm, y_norm, epochs, 2)
#
# """After the training, model is done, predict the next 200 points in a recursive fashion."""
# start_point = 1000
# next_points = 200
# prediction_norm = scale.inverse_transform(
#     TS_norm.predict_ahead_data_point(
#         cnn_norm, window_size, start_point, next_points))
# real_measurements = TS_orig.predict_ahead_data_point(
#     cnn_orig, window_size, start_point, next_points)
#
# plot(
#     [cnn_norm, cnn_orig],
#     [prediction_norm, real_measurements],
#     ["Cnn Normalization",
#      "CNN Original"],
#     2,
# )
