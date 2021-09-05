import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import matplotlib.pyplot as plt


"""
Author: Sheng Kuang, Yimin Yang
"""

# Load data
training_data_path = 'Xtrain.mat'
test_data_path = 'Xtest-1.mat'
raw_data = loadmat(training_data_path)['Xtrain']
test_data = loadmat(test_data_path)['Xtest']
print(raw_data.shape)

# Normalization
s = MinMaxScaler(feature_range=(0, 1))
s = s.fit(raw_data)
raw_data = s.transform(raw_data)

# Define hyper-parameters
WINDOW_SIZE = 70
PREDICT_SIZE = 1
TRAINING_DATA_PERCENTAGE = 0.99

LEARNING_RATE = 0.01
NUM_EPOCHS = 5000
INPUT_SIZE = 1
HIDDEN_SIZE = 8
NUM_LAYERS = 1
OUTPUT_SIZE = 1
SHUFFLE = False
PREDICT_STEPS = 200

# Pre-processing: scale the dataset
total_data_size = raw_data.shape[0] - WINDOW_SIZE - PREDICT_SIZE
input_data = np.zeros((total_data_size, WINDOW_SIZE))
output_data = np.zeros((total_data_size, PREDICT_SIZE))
for i_start in range(total_data_size):
    input_data[i_start, :] = raw_data[i_start:i_start + WINDOW_SIZE, :].T
    output_data[i_start, :] = raw_data[i_start + WINDOW_SIZE: i_start + WINDOW_SIZE + PREDICT_SIZE, :].T

# Shuffle data
if SHUFFLE:
    state = np.random.get_state()
    np.random.shuffle(input_data)
    np.random.set_state(state)
    np.random.shuffle(output_data)

# Split data into training and validation array
train_num = int(TRAINING_DATA_PERCENTAGE * total_data_size)
train_x = Variable(torch.Tensor(input_data[:train_num, :]))
train_y = Variable(torch.Tensor(output_data[:train_num, :]))
valid_x = Variable(torch.Tensor(input_data[train_num:, :]))
valid_y = Variable(torch.Tensor(output_data[train_num:, :]))
print('Training :', train_x.shape, train_y.shape, ', Validation :', valid_x.shape, valid_y.shape)

train_x = torch.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))  # records * seq_len * input_size
valid_x = torch.reshape(valid_x, (valid_x.shape[0], valid_x.shape[1], 1))  # records * seq_len * input_size
print('Training :', train_x.shape, train_y.shape, ', Validation :', valid_x.shape, valid_y.shape)


class Model1(nn.Module):
    """LSTM model class"""

    def __init__(self, v_input_size, v_hidden_size, v_num_layers, v_output_size, seq_len, batch_size):
        super(Model1, self).__init__()
        self.input_size = v_input_size
        self.hidden_size = v_hidden_size
        self.num_layers = v_num_layers
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, )
        self.fc = nn.Linear(self.hidden_size * self.seq_len, v_output_size)

    def forward(self, x):
        # feed forward
        # h_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size)  # num_layers, batch_size, hidden_size
        # c_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size)  # num_layers, batch_size, hidden_size
        # lstm_output, _ = self.lstm(x, (h_0, c_0))
        lstm_output, _ = self.lstm(x)
        out = self.fc(lstm_output.contiguous().view(-1, self.hidden_size * self.seq_len))
        return out

    def recursive_predict(self, x, steps):
        seq = np.zeros((steps + x.shape[0], 1))
        seq[:x.shape[0], :] = x
        for s_index in range(steps):
            input_seq = Variable(torch.Tensor(seq[s_index:s_index + x.shape[0], :].reshape(1, -1, 1)))
            output = self.forward(input_seq)
            seq[x.shape[0] + s_index, 0] = output[0, 0]
        return seq[x.shape[0]:, :]


def training(v_train_x, v_train_y, v_valid_x, v_valid_y, epochs, v_input_size, v_hidden_size, v_num_layers,
             v_output_size, lr):
    # Define model and loss function
    m = Model1(v_input_size, v_hidden_size, v_num_layers, v_output_size, seq_len=v_train_x.shape[1],
               batch_size=v_train_x.shape[0])
    criterion = nn.MSELoss()  # MSE
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)

    # Start training
    loss_log = np.zeros((epochs, 1))
    for e_index in range(epochs):
        outputs = m(v_train_x)
        optimizer.zero_grad()
        loss = criterion(outputs, v_train_y)
        loss.backward()
        optimizer.step()
        loss_log[e_index, 0] = loss.item()
        if e_index % 100 == 0:
            # Validation
            t_o = m(v_valid_x)
            t_l = criterion(t_o, v_valid_y)
            print("Epoch: {0}, Training loss: {1:1.8f}, Validation loss: {2:1.8f}".format(e_index, loss.item(),
                                                                                          t_l.item()))
    # Save model
    torch.save(m, 'LSTM.pkl')
    return m, loss_log


"""
============ =====================================================================
Prediction
------------ ---------------------------------------------------------------------
"""

# model, log = training(train_x, train_y, valid_x, valid_y, NUM_EPOCHS, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE,
#                  LEARNING_RATE)
# prediction = model.recursive_predict(raw_data[-WINDOW_SIZE:, :], PREDICT_STEPS)

model = torch.load('LSTM_final.pkl')
prediction = model.recursive_predict(raw_data[-40:, :], PREDICT_STEPS)
criterion = nn.MSELoss()

# Original
inverse_prediction = s.inverse_transform(prediction)
test_MSE = criterion(Variable(torch.Tensor(test_data)), Variable(torch.Tensor(inverse_prediction))).item()
print('Test MSE(Original): {}'.format(test_MSE))
print('Test MAE(Original): {}'.format(np.average(np.abs(test_data - inverse_prediction))))
# # Normalized
# test_MSE_n = criterion(Variable(torch.Tensor(s.transform(test_data))), Variable(torch.Tensor(prediction))).item()
# print('Test MSE(Normalized): {}'.format(test_MSE_n))


# plot predictions
fig = plt.figure(figsize=(8, 2))
plt.plot(s.inverse_transform(raw_data), label='Real data')

plt_x = np.array(range(0, PREDICT_STEPS + 1, 1)) + raw_data.shape[0] - 1
plt.plot(plt_x, s.inverse_transform(np.vstack((raw_data[-1, :].reshape((1, 1)), prediction))), label='Prediction')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(loc='upper right', ncol=3, fontsize=8)
plt.show()

# comparison
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2))
ax1.plot(s.inverse_transform(prediction), label='Prediction')
ax1.set_ylabel('Amplitude')
ax1.set_ylim(-20, 250)
# ax1.legend(loc='lower right')
ax1.grid()
ax1.set_title('Predictions')
ax2.plot(test_data, label='Test data')
ax2.set_ylim(-20, 250)
# ax2.legend(loc='lower right')
ax2.grid()
ax2.set_title('Test data')
ax3.plot(np.abs(s.inverse_transform(prediction) - test_data), label='Diff')
ax3.set_ylim(-20, 250)
# ax3.legend(loc='lower right')
ax3.grid()
ax3.set_title('Absolute error')

# ax4.plot(np.abs(s.inverse_transform(prediction) - test_data), label='Diff')
# ax4.set_ylim(-20, 250)
# # ax3.legend(loc='lower right')
# ax4.grid()
# ax4.set_title('Absolute error')

plt.show()


"""
============ =====================================================================
Fine-tuning: hidden_size
------------ ---------------------------------------------------------------------
"""
# num_exps = 10
# logs = np.zeros((NUM_EPOCHS, num_exps))
#
# for i in range(num_exps):
#     hidden_size = (i + 1) * 2
#     model, log = training(train_x, train_y, valid_x, valid_y, NUM_EPOCHS, INPUT_SIZE, hidden_size, NUM_LAYERS, OUTPUT_SIZE, LEARNING_RATE)
#     logs[:, i] = log[:, 0]
# # plot
# fig = plt.figure(figsize=(8, 3))
# for i in range(num_exps):
#     plt.plot(logs[:, i], label='Hidden size={}'.format((i + 1) * 2))
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
#
# for i in range(num_exps):
#     window_size = (i + 1) * 10
#
#     total_data_size = raw_data.shape[0] - window_size - PREDICT_SIZE
#     input_data = np.zeros((total_data_size, window_size))
#     output_data = np.zeros((total_data_size, PREDICT_SIZE))
#     for i_start in range(total_data_size):
#         input_data[i_start, :] = raw_data[i_start:i_start + window_size, :].T
#         output_data[i_start, :] = raw_data[i_start + window_size: i_start + window_size + PREDICT_SIZE, :].T
#
#     # Split data into training and validation array
#     train_num = int(TRAINING_DATA_PERCENTAGE * total_data_size)
#     train_x = Variable(torch.Tensor(input_data[:train_num, :]))
#     train_y = Variable(torch.Tensor(output_data[:train_num, :]))
#     valid_x = Variable(torch.Tensor(input_data[train_num:, :]))
#     valid_y = Variable(torch.Tensor(output_data[train_num:, :]))
#     print('Training :', train_x.shape, train_y.shape, ', Validation :', valid_x.shape, valid_y.shape)
#
#     train_x = torch.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))  # records * seq_len * input_size
#     valid_x = torch.reshape(valid_x, (valid_x.shape[0], valid_x.shape[1], 1))  # records * seq_len * input_size
#     print('Training :', train_x.shape, train_y.shape, ', Validation :', valid_x.shape, valid_y.shape)
#
#     model, log = training(train_x, train_y, valid_x, valid_y, NUM_EPOCHS, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, LEARNING_RATE)
#     logs[:, i] = log[:, 0]
# # plot
# fig = plt.figure(figsize=(8, 3))
# for i in range(num_exps):
#     plt.plot(logs[:, i], label='Window size={}'.format((i + 1) * 10))
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
# num_exps = 5
# logs = np.zeros((NUM_EPOCHS, num_exps))
#
# for i in range(num_exps):
#     learning_rate = 0.1 ** (i + 1)
#     model, log = training(train_x, train_y, valid_x, valid_y, NUM_EPOCHS, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, learning_rate)
#     logs[:, i] = log[:, 0]
# # plot
# fig = plt.figure(figsize=(8, 3))
# for i in range(num_exps):
#     plt.plot(logs[:, i], label='Learning rate={}'.format(np.round(0.1 ** (i + 1), i + 1)))
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
Fine-tuning: num layers
------------ ---------------------------------------------------------------------
"""
# num_exps = 4
# logs = np.zeros((NUM_EPOCHS, num_exps))
#
# for i in range(num_exps):
#     num_layers = i + 1
#     model, log = training(train_x, train_y, valid_x, valid_y, NUM_EPOCHS, INPUT_SIZE, HIDDEN_SIZE, num_layers, OUTPUT_SIZE, LEARNING_RATE)
#     logs[:, i] = log[:, 0]
# # plot
# fig = plt.figure(figsize=(8, 3))
# for i in range(num_exps):
#     plt.plot(logs[:, i], label='Num layers={}'.format(i + 1))
#
# plt.legend(loc='upper right', ncol=3, fontsize=8)
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# # plt.ylim(0, 0.05)
# plt.yscale('log')
# plt.title('Experiment 4: Num layers')
# plt.show()
