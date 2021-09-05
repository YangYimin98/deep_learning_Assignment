import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from datetime import datetime
import matplotlib.pyplot as plt
from copy import deepcopy

"""
Author: Sheng Kuang, Yimin Yang
"""

WINDOW_SIZE = 100  # 80
PREDICT_SIZE = 1

BATCH_SIZE = 2000
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
INPUT_SIZE = 6
HIDDEN_SIZE = 5  # 50
NUM_LAYERS = 1
OUTPUT_SIZE = 1
PREDICT_STEPS = 168
T_UNIT = 0.1


def normalization(raw_data, window_size, predict_size, test_length=168):
    # Normalization
    city_feature = np.ones((raw_data.shape[0], raw_data.shape[1], 1))
    for ic in range(raw_data.shape[1]):
        city_feature[:, ic, :] = ic
    raw_data = np.concatenate((raw_data, city_feature), axis=2)

    scale = MinMaxScaler(feature_range=(0, 1))
    raw_shape = raw_data.shape
    print('Raw data size: ', raw_shape)
    location_size = raw_shape[1]
    feature_size = raw_shape[2]
    raw_data = raw_data.reshape((-1, raw_shape[2]))
    scale = scale.fit(raw_data)
    raw_data = scale.transform(raw_data)
    raw_data = raw_data.reshape(raw_shape)

    # Pre-processing: scale the dataset
    total_data_size = raw_data.shape[0] - window_size - predict_size
    # ** total_data_size * window_size * (feature)
    input_data = np.zeros((location_size * total_data_size, window_size, feature_size))
    # ** total_data_size * (temperature)
    output_data = np.zeros((location_size * total_data_size, OUTPUT_SIZE))
    for i_start in range(total_data_size):
        for l_start in range(location_size):
            input_data[i_start + l_start * total_data_size, :] = raw_data[i_start:i_start + window_size, l_start, :].reshape((window_size, -1))
            output_data[i_start + l_start * total_data_size, :] = raw_data[i_start + window_size + 1: i_start + window_size + predict_size + 1, l_start, 2].reshape(1)
    print('Input data size :', input_data.shape, ', Output data size', output_data.shape)

    # Split data into training and validation array
    train_num = total_data_size - 2 * test_length

    t_train_x_l = []
    t_train_y_l = []
    t_valid_x_l = []
    t_valid_y_l = []
    t_test_x_l = []
    t_test_y_l = []
    for l_start in range(location_size):
        t_train_x_l.append(input_data[l_start * total_data_size:l_start * total_data_size + train_num, :, :])  # records * seq_len * input_size
        t_train_y_l.append(output_data[l_start * total_data_size:l_start * total_data_size + train_num, :])
        t_valid_x_l.append(input_data[l_start * total_data_size + train_num:l_start * total_data_size + train_num + test_length, :, :])
        t_valid_y_l.append(output_data[l_start * total_data_size + train_num:l_start * total_data_size + train_num + test_length, :])
        t_test_x_l.append(input_data[l_start * total_data_size + train_num + test_length:l_start * total_data_size + train_num + 2 * test_length, :, :])
        t_test_y_l.append(output_data[l_start * total_data_size + train_num + test_length:l_start * total_data_size + train_num + 2 * test_length, :])
    train_x = Variable(torch.Tensor(np.concatenate(t_train_x_l, axis=0)))
    train_y = Variable(torch.Tensor(np.concatenate(t_train_y_l, axis=0)))
    valid_x = Variable(torch.Tensor(np.concatenate(t_valid_x_l, axis=0)))
    valid_y = Variable(torch.Tensor(np.concatenate(t_valid_y_l, axis=0)))
    test_x = Variable(torch.Tensor(np.concatenate(t_test_x_l, axis=0)))
    test_y = Variable(torch.Tensor(np.concatenate(t_test_y_l, axis=0)))

    print('Training :', train_x.shape, train_y.shape, ', Validation :', valid_x.shape, valid_y.shape, ', Testing :',
          test_x.shape, test_y.shape)
    return train_x, train_y, valid_x, valid_y, test_x, test_y, scale, raw_data


class Model1(nn.Module):
    """LSTM model class"""

    def __init__(self, v_input_size, v_hidden_size, v_num_layers, v_output_size, seq_len, num_attn_head=0, opt='Adam'):
        super(Model1, self).__init__()
        self.input_size = v_input_size
        self.hidden_size = v_hidden_size
        self.num_layers = v_num_layers
        self.seq_len = seq_len
        self.training_x = None
        self.training_y = None
        self.validation_x = None
        self.validation_y = None
        self.num_attn_head = num_attn_head
        self.opt = opt

        if not num_attn_head:
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
            self.fc = nn.Linear(self.hidden_size * self.seq_len, v_output_size)
        if num_attn_head:
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
            self.attn = nn.MultiheadAttention(self.hidden_size, num_attn_head)
            self.fc = nn.Linear(self.hidden_size * self.seq_len, v_output_size)

    def forward(self, x):
        # feed forward
        # h_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size)  # num_layers, batch_size, hidden_size
        # c_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size)  # num_layers, batch_size, hidden_size
        # lstm_output, _ = self.lstm(x, (h_0, c_0))
        if not self.num_attn_head:
            lstm_output, _ = self.lstm(x)
            out = self.fc(lstm_output.contiguous().view(-1, self.hidden_size * self.seq_len))
            return out
        else:
            lstm_output, _ = self.lstm(x)
            self.attn = 0
            out = self.fc(lstm_output.contiguous().view(-1, self.hidden_size * self.seq_len))
            return out

    def set_training_data(self, training_x, training_y, validation_x=None, validation_y=None):
        self.training_x = training_x
        self.training_y = training_y
        self.validation_x = validation_x
        self.validation_y = validation_y

    def train(self, epochs, batch_size, lr):
        criterion = nn.MSELoss()  # MSE
        if self.opt == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif self.opt == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif self.opt == 'SGD+Momentum':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.5)
        elif self.opt == 'SGD+Nesterov':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.5, nesterov=True)
        elif self.opt == 'AdaGrad':
            optimizer = torch.optim.Adagrad(self.parameters(), lr=lr)
        elif self.opt == 'RMSProp':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        num_batches = np.int(np.floor(self.training_x.shape[0] / batch_size))

        # Start training
        loss_log = np.zeros((epochs, 1))
        for e_index in range(epochs):
            for b_index in range(num_batches):
                outputs = self.forward(self.training_x[b_index * batch_size:b_index * (batch_size + 1), :, :])
                optimizer.zero_grad()
                loss = criterion(outputs, self.training_y[b_index * batch_size:b_index * (batch_size + 1), :])
                loss.backward()
                optimizer.step()
                loss_log[e_index, 0] = loss.item()
            # if e_index % 10 == 0:
            # Validation
            t_o = self.forward(self.validation_x)
            t_l = criterion(t_o, self.validation_y)
            print(datetime.now(), "Epoch: {0}, Training loss: {1:1.8f}, Validation loss: {2:1.8f}".format(e_index, loss.item(),
                                                                                          t_l.item()))
        # Save model
        # torch.save(self.modules(), 'LSTM.pkl')
        return loss_log

    def recursive_predict(self, x, y, steps):
        """

        :param x: 168 * window_size * input_size
        :param y: 168 * 4/1, ground truth
        :param steps: recursively prediction steps
        :return: predictions
        """

        seq = np.zeros((x.shape[1] + x.shape[0], x.shape[2]))  # [window_size + 168, input_size]
        seq[:x.shape[1], :] = x[0, :, :]

        predict_y = np.zeros(y.shape)
        for s_index in range(steps):

            input_seq = x[s_index, :, :].reshape(1, x.shape[1], x.shape[2])  # 1 * window_size * input_size
            input_seq[0, :, 2] = Variable(torch.Tensor(seq[s_index:s_index + x.shape[1], 2]))
            # input_seq = Variable(torch.Tensor(input_seq))
            output = self.forward(input_seq)
            seq[s_index + x.shape[1], 2] = output[0, 0]
            if s_index != steps - 1:
                seq[s_index + x.shape[1], :2] = x[s_index + 1, -1, :2]
                seq[s_index + x.shape[1], 3:] = x[s_index + 1, -1, 3:]
            predict_y[s_index, :] = output[0, 0].detach().numpy()
        return predict_y


# Training
training_data_path = 'data.mat'
raw_dataset = loadmat(training_data_path)['X']


tr_x, tr_y, v_x, v_y, te_x, te_y, s, normalized_data = normalization(raw_dataset, WINDOW_SIZE, PREDICT_SIZE)  # seq_len(window_size), predict_size
m = Model1(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, WINDOW_SIZE)  # input_size, hidden_size, num_layer, output_size, seq_len(window_size)
m.set_training_data(tr_x, tr_y, v_x, v_y)
m.train(NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)  # epoch, batch_size, learning_rate
torch.save(m, 'LSTM.pkl')

real_max = s.data_max_[2]
real_min = s.data_min_[2]

for city_index in range(4):
    # real_data_normalized = normalized_data[-PREDICT_STEPS:, city_index, 2]
    real_data = raw_dataset[-PREDICT_STEPS:, city_index, 2]

    prediction_normalized = m.recursive_predict(te_x[city_index * PREDICT_STEPS:(city_index + 1) * PREDICT_STEPS, :, :], te_y[city_index * PREDICT_STEPS:(city_index + 1) * PREDICT_STEPS, :], PREDICT_STEPS)
    prediction = prediction_normalized.reshape(PREDICT_STEPS) * (real_max-real_min) + real_min
    MAE = np.average(np.abs(prediction - real_data) * T_UNIT)
    print('City {} MAE is :'.format(city_index), MAE)

# Plot MAE
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.plot(prediction * T_UNIT, label='Prediction')
ax1.set_ylabel('Temperature(C)')
ax1.set_xlabel('Time index (hour)')
ax1.set_ylim(-5, 10)
ax1.grid()
ax1.set_title('Real and Prediction data')
ax1.plot(real_data * T_UNIT, label='Real data')
ax1.legend(loc='lower right', fontsize=12)

ax2.plot(np.abs(prediction - real_data) * T_UNIT, label='Diff')
# ax3.legend(loc='lower right')
ax2.grid()
ax2.set_ylim(0, 6)
ax2.set_xlabel('Time index (hour)')
ax2.set_title('Absolute error')
plt.show()

# --------------report plot---------------
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

ax1.plot(LSTM5 * T_UNIT, label='Prediction')
ax1.set_ylabel('Temperature(C)')
ax1.set_xlabel('Time index (hour)')
ax1.set_ylim(-5, 10)
ax1.grid()
ax1.set_title('LSTM-5')
ax1.plot(real_data * T_UNIT, label='Real data')
ax1.legend(loc='lower right', fontsize=12)

ax2.plot(prediction * T_UNIT, label='Prediction')
# ax2.set_ylabel('Temperature(C)')
ax2.set_xlabel('Time index (hour)')
ax2.set_ylim(-5, 10)
ax2.grid()
ax2.set_title('LSTM-6')
ax2.plot(real_data * T_UNIT, label='Real data')
ax2.legend(loc='lower right', fontsize=12)

ax3.plot(yym_cnn, label='Prediction')
# ax3.set_ylabel('Temperature(C)')
ax3.set_xlabel('Time index (hour)')
ax3.set_ylim(-5, 10)
ax3.grid()
ax3.set_title('CNN')
ax3.plot(real_data * T_UNIT, label='Real data')
ax3.legend(loc='lower right', fontsize=12)
plt.show()

"""
============ =====================================================================
Fine-tuning: hidden_size
------------ ---------------------------------------------------------------------
"""
# num_exps = 10
# logs = np.zeros((NUM_EPOCHS, num_exps))
# for i in range(num_exps):
#     hidden_size = (i + 2) * 2
#     tr_x, tr_y, v_x, v_y, te_x, te_y, s, normalized_data = normalization(raw_dataset, WINDOW_SIZE, PREDICT_SIZE)  # seq_len(window_size), predict_size
#     m = Model1(INPUT_SIZE, hidden_size, NUM_LAYERS, OUTPUT_SIZE, WINDOW_SIZE)  # input_size, hidden_size, num_layer, output_size, seq_len(window_size)
#     m.set_training_data(tr_x, tr_y, v_x, v_y)
#     log = m.train(NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)  # epoch, batch_size, learning_rate
#     logs[:, i] = log[:, 0]
# # plot
# fig = plt.figure(figsize=(5, 3))
# for i in range(6):
#     plt.plot(logs[:, i], label='H={}'.format((i + 2) * 2))
#
# plt.legend(loc='upper right', ncol=3, fontsize=8)
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# # plt.ylim(0, 0.05)
# plt.yscale('log')
# plt.title('Experiment 1: Hidden size')
# plt.show()

"""
============ =====================================================================
Fine-tuning: window_size
------------ ---------------------------------------------------------------------
"""
# num_exps = 10
# logs = np.zeros((NUM_EPOCHS, num_exps))
# for i in range(num_exps):
#     window_size = (i + 2) * 10
#     tr_x, tr_y, v_x, v_y, te_x, te_y, s, normalized_data = normalization(raw_dataset, window_size, PREDICT_SIZE)  # seq_len(window_size), predict_size
#     m = Model1(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, window_size)  # input_size, hidden_size, num_layer, output_size, seq_len(window_size)
#     m.set_training_data(tr_x, tr_y, v_x, v_y)
#     log = m.train(NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)  # epoch, batch_size, learning_rate
#     logs[:, i] = log[:, 0]
# # plot
# fig = plt.figure(figsize=(5, 3))
# for i in range(num_exps):
#     plt.plot(logs[:, i], label='W={}'.format((i + 2) * 10))
#
# plt.legend(loc='upper right', ncol=3, fontsize=8)
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# # plt.ylim(0, 0.05)
# plt.yscale('log')
# plt.title('Experiment 2: Window size')
# plt.show()

"""
============ =====================================================================
Fine-tuning: learning rate
------------ ---------------------------------------------------------------------
"""
# num_exps = 4
# logs = np.zeros((NUM_EPOCHS, num_exps))
# for i in range(num_exps):
#     learning_rate = 0.1 ** (i + 1)
#     tr_x, tr_y, v_x, v_y, te_x, te_y, s, normalized_data = normalization(raw_dataset, WINDOW_SIZE, PREDICT_SIZE)  # seq_len(window_size), predict_size
#     m = Model1(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, WINDOW_SIZE)  # input_size, hidden_size, num_layer, output_size, seq_len(window_size)
#     m.set_training_data(tr_x, tr_y, v_x, v_y)
#     log = m.train(NUM_EPOCHS, BATCH_SIZE, learning_rate)  # epoch, batch_size, learning_rate
#     logs[:, i] = log[:, 0]
# # plot
# fig = plt.figure(figsize=(5, 3))
# for i in range(num_exps):
#     plt.plot(logs[:, i], label='Lr={}'.format(np.round(0.1 ** (i + 1), i + 1)))
#
# plt.legend(loc='upper right', ncol=3, fontsize=8)
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# # plt.ylim(0, 0.05)
# plt.yscale('log')
# plt.title('Experiment 3: Learning rate')
# plt.show()
"""
============ =====================================================================
Fine-tuning: batch_size
------------ ---------------------------------------------------------------------
"""
# num_exps = 4
# logs = np.zeros((NUM_EPOCHS, num_exps))
# batch_l = [100, 200, 500, 1000]
# for i in range(num_exps):
#     batch_size = batch_l[i]
#     tr_x, tr_y, v_x, v_y, te_x, te_y, s, normalized_data = normalization(raw_dataset, WINDOW_SIZE, PREDICT_SIZE)  # seq_len(window_size), predict_size
#     m = Model1(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, WINDOW_SIZE)  # input_size, hidden_size, num_layer, output_size, seq_len(window_size)
#     m.set_training_data(tr_x, tr_y, v_x, v_y)
#     log = m.train(NUM_EPOCHS, batch_size, LEARNING_RATE)  # epoch, batch_size, learning_rate
#     logs[:, i] = log[:, 0]
# # plot
# fig = plt.figure(figsize=(5, 3))
# for i in range(num_exps):
#     plt.plot(logs[:, i], label='W={}'.format(batch_l[i]))
#
# plt.legend(loc='upper right', ncol=3, fontsize=8)
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# # plt.ylim(0, 0.05)
# plt.yscale('log')
# plt.title('Experiment 4: Batch size')
# plt.show()

"""
============ =====================================================================
Fine-tuning: num_layers
------------ ---------------------------------------------------------------------
"""
# num_exps = 3
# logs = np.zeros((NUM_EPOCHS, num_exps))
# for i in range(num_exps):
#     num_layers = i + 1
#     tr_x, tr_y, v_x, v_y, te_x, te_y, s, normalized_data = normalization(raw_dataset, WINDOW_SIZE, PREDICT_SIZE)  # seq_len(window_size), predict_size
#     m = Model1(INPUT_SIZE, HIDDEN_SIZE, num_layers, OUTPUT_SIZE, WINDOW_SIZE)  # input_size, hidden_size, num_layer, output_size, seq_len(window_size)
#     m.set_training_data(tr_x, tr_y, v_x, v_y)
#     log = m.train(NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)  # epoch, batch_size, learning_rate
#     logs[:, i] = log[:, 0]
# # plot
# fig = plt.figure(figsize=(5, 3))
# for i in range(num_exps):
#     plt.plot(logs[:, i], label='W={}'.format(i + 1))
#
# plt.legend(loc='upper right', ncol=3, fontsize=8)
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# # plt.ylim(0, 0.05)
# plt.yscale('log')
# plt.title('Experiment 5: Num layers')
# plt.show()

"""
============ =====================================================================
Fine-tuning: Optimizer
------------ ---------------------------------------------------------------------
"""
# num_exps = 6
# logs = np.zeros((NUM_EPOCHS, num_exps))
# optimizer = ['SGD', 'SGD+Momentum', 'SGD+Nesterov', 'AdaGrad', 'RMSProp', 'Adam']
# for i in range(num_exps):
#     opt = optimizer[i]
#     tr_x, tr_y, v_x, v_y, te_x, te_y, s, normalized_data = normalization(raw_dataset, WINDOW_SIZE, PREDICT_SIZE)  # seq_len(window_size), predict_size
#     m = Model1(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, WINDOW_SIZE, opt=opt)  # input_size, hidden_size, num_layer, output_size, seq_len(window_size)
#     m.set_training_data(tr_x, tr_y, v_x, v_y)
#     log = m.train(NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)  # epoch, batch_size, learning_rate
#     logs[:, i] = log[:, 0]
# # plot
# fig = plt.figure(figsize=(5, 3))
# for i in [0, 3, 4, 5]:
#     plt.plot(logs[:, i], label='{}'.format(optimizer[i]))
#
# plt.legend(loc='upper right', ncol=3, fontsize=8)
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# # plt.ylim(0, 0.05)
# plt.yscale('log')
# plt.title('Experiment 6: Optimizer')
# plt.show()
"""
============ =====================================================================
Feature occlusion analysis (Spatial) 
------------ ---------------------------------------------------------------------
"""
# Real data
# real_data_normalized = normalized_data[-168:, 3, 2]
# real_data = raw_dataset[-168:, 3, 2]
real_max = np.max(raw_dataset[:, :, 2])
real_min = np.min(raw_dataset[:, :, 2])

# Prediction
prediction_normalized = m.recursive_predict(te_x[-168:, :, :], te_y[-168:, :], 168).reshape(PREDICT_STEPS)
prediction = prediction_normalized * (real_max-real_min) + real_min
MSE = np.average(np.square(0.1 * (prediction - real_data)))
print('MSE is :', MSE)

# spatial analysis  Mask features
exp_times = 10
delta_MSE_features = np.zeros((exp_times, 6, 4))
for t in range(exp_times):  # times
    for i in range(6):  # features
        for city_index in range(4):  # cities
            # reference MSE
            real_data = raw_dataset[-PREDICT_STEPS:, city_index, 2]
            prediction_normalized = m.recursive_predict(te_x[city_index * PREDICT_STEPS:(city_index + 1) * PREDICT_STEPS, :, :], te_y[city_index * PREDICT_STEPS:(city_index + 1) * PREDICT_STEPS, :], PREDICT_STEPS).reshape(PREDICT_STEPS)
            prediction = prediction_normalized * (real_max - real_min) + real_min
            MSE = np.average(np.square(0.1 * (prediction - real_data)))

            # masked MSE
            dte_x = deepcopy(te_x[city_index * PREDICT_STEPS:(city_index + 1) * PREDICT_STEPS:, :, :])
            dte_x[:, :, i] = Variable(torch.Tensor(np.random.random_sample((PREDICT_STEPS, WINDOW_SIZE))))
            masked_prediction_normalized = m.recursive_predict(dte_x, te_y[city_index * PREDICT_STEPS:(city_index + 1) * PREDICT_STEPS, :], PREDICT_STEPS).reshape(PREDICT_STEPS)
            masked_prediction = masked_prediction_normalized * (real_max - real_min) + real_min
            masked_MSE = np.average(np.square(0.1 * (masked_prediction - real_data)))
            delta_MSE = (masked_MSE - MSE) / MSE
            # print('MSE is :', MSE, 'masked_MSE is :', masked_MSE, 'delta_MSE is : ', delta_MSE)
            delta_MSE_features[t, i, city_index] = delta_MSE
avg_delta_MSE = np.average(delta_MSE_features, axis=0)
print(avg_delta_MSE)


# visualization the spatial analysis
x_city = np.arange(0, 5, 1)  # len = 11
y_feature = np.arange(0, 6, 1)  # len = 7
Z_mse = avg_delta_MSE[:5, :]
fig, ax = plt.subplots(figsize=(4, 3))
cs = ax.pcolormesh(x_city, y_feature, Z_mse)
ax.set_xticks(np.arange(0.5, 4, 1))
ax.set_xticklabels(('City 0', 'City 1', 'City 2', 'City 3'))
ax.set_yticks(np.arange(0.5, 5, 1))
ax.set_yticklabels(('Wind S.', 'Wind D.', 'Temp.', 'Dew P.', 'Air P.'))
fig.colorbar(cs)
plt.xlabel('Cities')
plt.ylabel('Features')
fig.show()

"""
============ =====================================================================
Occlusion analysis (Temporal) single feature masked with sliding window
------------ ---------------------------------------------------------------------
"""

# Temporal analysis  Mask features
exp_times = 1
mask_len = 10
windows = np.int(WINDOW_SIZE / mask_len)
features = 5
cities = 4
delta_MSE_features = np.zeros((exp_times, features, cities, windows))
for t in range(exp_times):  # times
    for i in range(features):  # features
        for city_index in range(cities):  # cities

            # reference MSE
            real_data = raw_dataset[-PREDICT_STEPS:, city_index, 2]
            prediction_normalized = m.recursive_predict(
                te_x[city_index * PREDICT_STEPS:(city_index + 1) * PREDICT_STEPS, :, :],
                te_y[city_index * PREDICT_STEPS:(city_index + 1) * PREDICT_STEPS, :], PREDICT_STEPS).reshape(
                PREDICT_STEPS)
            prediction = prediction_normalized * (real_max - real_min) + real_min
            MSE = np.average(np.square(0.1 * (prediction - real_data)))

            for w in range(windows):
                # masked MSE
                dte_x = deepcopy(te_x[city_index * PREDICT_STEPS:(city_index + 1) * PREDICT_STEPS:, :, :])
                for mask_i in range(PREDICT_STEPS):
                    d2_s = w * mask_len - mask_i if w * mask_len - mask_i >= 0 else 0
                    d2_e = (w + 1) * mask_len - mask_i if (w + 1) * mask_len - mask_i >= 0 else 0
                    if d2_s == 0 and d2_e == 0:
                        pass
                    else:
                        dte_x[mask_i, d2_s:d2_e, i] = Variable(torch.Tensor(np.random.random_sample((d2_e-d2_s))))

                masked_prediction_normalized = m.recursive_predict(dte_x, te_y[city_index * PREDICT_STEPS:(city_index + 1) * PREDICT_STEPS, :], PREDICT_STEPS).reshape(PREDICT_STEPS)
                masked_prediction = masked_prediction_normalized * (real_max - real_min) + real_min
                masked_MSE = np.average(np.square(0.1 * (masked_prediction - real_data)))
                delta_MSE = (masked_MSE - MSE) / MSE
                # print('MSE is :', MSE, 'masked_MSE is :', masked_MSE, 'delta_MSE is : ', delta_MSE)
                delta_MSE_features[t, i, city_index, w] = delta_MSE
avg_delta_MSE = np.average(delta_MSE_features, axis=(0, 2))
print(avg_delta_MSE)


# visualization the temporal analysis
x_window = np.arange(0, windows + 1, 1)
y_feature = np.arange(0, features + 1, 1)
Z_mse = avg_delta_MSE
fig, ax = plt.subplots(figsize=(4, 3))
cs = ax.pcolormesh(x_window, y_feature, Z_mse)
ax.set_xticks(np.arange(0.5, windows, 1))
ax.set_xticklabels(np.arange(1, windows + 1, 1))
ax.set_yticks(np.arange(0.5, features, 1))
ax.set_yticklabels(('Wind S.', 'Wind D.', 'Temp.', 'Dew P.', 'Air P.'))
fig.colorbar(cs)
plt.xlabel('Lags Index')
plt.ylabel('Features')
plt.title("Mask size '{}*1'".format(mask_len))
fig.show()

"""
============ =====================================================================
Occlusion analysis (Temporal) whole feature masked with sliding window
------------ ---------------------------------------------------------------------
"""
# Temporal analysis  Mask features
exp_times = 1
mask_len = 10
windows = np.int(WINDOW_SIZE / mask_len)
features = 5
cities = 4
delta_MSE_features = np.zeros((exp_times, cities, windows))
for t in range(exp_times):  # times
    for city_index in range(cities):  # cities

        # reference MSE
        real_data = raw_dataset[-PREDICT_STEPS:, city_index, 2]
        prediction_normalized = m.recursive_predict(
            te_x[city_index * PREDICT_STEPS:(city_index + 1) * PREDICT_STEPS, :, :],
            te_y[city_index * PREDICT_STEPS:(city_index + 1) * PREDICT_STEPS, :], PREDICT_STEPS).reshape(
            PREDICT_STEPS)
        prediction = prediction_normalized * (real_max - real_min) + real_min
        MSE = np.average(np.square(0.1 * (prediction - real_data)))

        for w in range(windows):
            # masked MSE
            dte_x = deepcopy(te_x[city_index * PREDICT_STEPS:(city_index + 1) * PREDICT_STEPS:, :, :])
            for mask_i in range(PREDICT_STEPS):
                d2_s = w * mask_len - mask_i if w * mask_len - mask_i >= 0 else 0
                d2_e = (w + 1) * mask_len - mask_i if (w + 1) * mask_len - mask_i >= 0 else 0
                if d2_s == 0 and d2_e == 0:
                    pass
                else:
                    dte_x[mask_i, d2_s:d2_e, :] = Variable(torch.Tensor(np.random.random_sample((d2_e-d2_s, 6))))

            masked_prediction_normalized = m.recursive_predict(dte_x, te_y[city_index * PREDICT_STEPS:(city_index + 1) * PREDICT_STEPS, :], PREDICT_STEPS).reshape(PREDICT_STEPS)
            masked_prediction = masked_prediction_normalized * (real_max - real_min) + real_min
            masked_MSE = np.average(np.square(0.1 * (masked_prediction - real_data)))
            delta_MSE = (masked_MSE - MSE) / MSE
            # print('MSE is :', MSE, 'masked_MSE is :', masked_MSE, 'delta_MSE is : ', delta_MSE)
            delta_MSE_features[t, city_index, w] = delta_MSE
avg_delta_MSE = np.average(delta_MSE_features, axis=0)
print(avg_delta_MSE)


# visualization the temporal analysis
x_window = np.arange(0, windows + 1, 1)
y_city = np.arange(0, cities + 1, 1)
Z_mse = avg_delta_MSE
fig, ax = plt.subplots(figsize=(4, 3))
cs = ax.pcolormesh(x_window, y_city, Z_mse)
ax.set_xticks(np.arange(0.5, windows, 1))
ax.set_xticklabels(np.arange(1, windows + 1, 1))
ax.set_yticks(np.arange(0.5, cities, 1))
ax.set_yticklabels(('City 0', 'City 1', 'City 2', 'City 3'))
fig.colorbar(cs)
plt.xlabel('Lags Index')
plt.ylabel('Features')
plt.title("Mask size '{}*6'".format(mask_len))
fig.show()
