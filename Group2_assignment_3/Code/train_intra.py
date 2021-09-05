# from datetime import datetime
from read_file import *
import matplotlib.pyplot as plt

from model import *

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.autograd import Variable


"""Hyperparameters"""
INPUT_CHANNEL = 248
INPUT_LENGTH = 2226
LATENT_CHANNEL = 32
NUM_CLASSES = 4
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 2
POOLING = 16
DROPOUT = 0.0
EPOCHS = 500
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
L1_WEIGHT = 0.001

"""Training"""
m_intra_train = get_subject_data_set(p1='Intra', p2='train',
                                     subject_state=['rest', 'task_motor', 'task_story_math', 'task_working_memory'],
                                     subject_id=None, resampling_scale=16)
m_intra_test = get_subject_data_set(p1='Intra', p2='test',
                                    subject_state=['rest', 'task_motor', 'task_story_math', 'task_working_memory'],
                                    subject_id=None, resampling_scale=16)


"""Training data set"""
train_size = len(m_intra_train)
batch_size = 8
num_batches = np.int(train_size / batch_size)
train_x = np.zeros((num_batches, batch_size, m_intra_train[0][0].shape[0], m_intra_train[0][0].shape[1]))
train_y = np.zeros((num_batches, batch_size, 1))
for b_index in range(num_batches):
    for s_index in range(batch_size):
        train_x[b_index, s_index, :, :] = m_intra_train[b_index * batch_size + s_index][0]
        train_y[b_index, s_index, :] = m_intra_train[b_index * batch_size + s_index][1]
train_x = Variable(torch.Tensor(train_x))
train_y = Variable(torch.Tensor(train_y)).long()
print('Training data set x size: ', train_x.shape)
print('Training data set y size: ', train_y.shape)

"""Testing data set"""
test_size = len(m_intra_test)
test_x = np.zeros((test_size, m_intra_test[0][0].shape[0], m_intra_test[0][0].shape[1]))
test_y = np.zeros((test_size, 1))
for t_index in range(test_size):
    test_x[t_index, :, :] = m_intra_test[t_index][0]
    test_y[t_index, :] = m_intra_test[t_index][1]
test_x = Variable(torch.Tensor(test_x))
test_y = Variable(torch.Tensor(test_y)).long()
print('Testing data set x size: ', test_x.shape)
print('Testing data set y size: ', test_y.shape)

"""Training"""
model = ModelCNN(input_channel=INPUT_CHANNEL, input_len=INPUT_LENGTH, latent_channel=LATENT_CHANNEL, k=KERNEL_SIZE,
                 s=STRIDE, p=PADDING, pool=POOLING, dropout=DROPOUT, num_classes=NUM_CLASSES)
log = model.training_model(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, l1_weight=L1_WEIGHT)

"""Prediction"""
predict = torch.argmax(model(test_x), dim=1)
prec_t = torch.eq(predict, test_y.T)
precision = torch.count_nonzero(prec_t) / prec_t.shape[1]
print('Test Precision :', precision.item())
# torch.save(model, 'intra.pkl')


"""plot MSE vs. Epochs"""
#
# fig = plt.figure(figsize=(5, 3))
# for i in range(log.shape[1]):
#     plt.plot(log[:, i])
# plt.xlabel('Epochs')
# plt.ylabel('Error')
# plt.yscale('log')
# # plt.title('Experiment 1: Hidden size')
# plt.show()

"""
============ =====================================================================
Fine-tuning: Kernel Size
------------ ---------------------------------------------------------------------
"""
# num_exps = 5
# logs = np.zeros((EPOCHS, num_exps))
# prec_l = np.zeros(num_exps)
# ks = [3, 5, 7, 9, 11]
# for i in range(num_exps):
#     kernel_size = ks[i]
#     """Training"""
#     model = ModelCNN(input_channel=INPUT_CHANNEL, input_len=INPUT_LENGTH, latent_channel=LATENT_CHANNEL, k=kernel_size,
#                      s=STRIDE, p=PADDING, pool=POOLING, dropout=DROPOUT, num_classes=NUM_CLASSES)
#     log = model.training_model(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
#     logs[:, i] = log[:, 0]
#
#     """Prediction"""
#     predict = torch.argmax(model(test_x), dim=1)
#     prec_t = torch.eq(predict, test_y.T)
#     precision = torch.count_nonzero(prec_t) / prec_t.shape[1]
#     prec_l[i] = (precision.item())
#     print('Test Precision :', precision.item())
#
# print('Precisions:', prec_l)
# print('Error:', logs[-1, :])
# """Plot"""
# fig = plt.figure(figsize=(5, 3))
# for i in range(num_exps):
#     plt.plot(logs[:, i], label='K={}'.format(ks[i]))
#
# plt.legend(loc='upper right', ncol=3, fontsize=8)
# plt.xlabel('Epochs')
# plt.ylabel('CrossEntropyLoss')
# plt.yscale('log')
# plt.title('Experiment 1: Kernel Size')
# plt.show()


"""
============ =====================================================================
Fine-tuning: LATENT_CHANNEL
------------ ---------------------------------------------------------------------
"""

# num_exps = 5
# logs = np.zeros((EPOCHS, num_exps))
# prec_l = np.zeros(num_exps)
# ks = [4, 8, 16, 32, 64]
# for i in range(num_exps):
#     latent_channel = ks[i]
#     """Training"""
#     model = ModelCNN(input_channel=INPUT_CHANNEL, input_len=INPUT_LENGTH, latent_channel=latent_channel, k=KERNEL_SIZE,
#                      s=STRIDE, p=PADDING, pool=POOLING, dropout=DROPOUT, num_classes=NUM_CLASSES)
#     log = model.training_model(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
#     logs[:, i] = log[:, 0]
#
#     """Prediction"""
#     predict = torch.argmax(model(test_x), dim=1)
#     prec_t = torch.eq(predict, test_y.T)
#     precision = torch.count_nonzero(prec_t) / prec_t.shape[1]
#     prec_l[i] = (precision.item())
#     print('Test Precision :', precision.item())
#
# print('Precisions:', prec_l)
# print('Error:', logs[-1, :])
# """Plot"""
# fig = plt.figure(figsize=(5, 3))
# for i in range(num_exps):
#     plt.plot(logs[:, i], label='K={}'.format(ks[i]))
#
# plt.legend(loc='upper right', ncol=3, fontsize=8)
# plt.xlabel('Epochs')
# plt.ylabel('CrossEntropyLoss')
# plt.yscale('log')
# plt.title('Experiment 2: Latent Channel')
# plt.show()

"""
============ =====================================================================
Fine-tuning: STRIDE
------------ ---------------------------------------------------------------------
"""

# num_exps = 4
# logs = np.zeros((EPOCHS, num_exps))
# prec_l = np.zeros(num_exps)
# ks = [1, 2, 3, 4]
# for i in range(num_exps):
#     stride = ks[i]
#     """Training"""
#     model = ModelCNN(input_channel=INPUT_CHANNEL, input_len=INPUT_LENGTH, latent_channel=LATENT_CHANNEL, k=KERNEL_SIZE,
#                      s=stride, p=PADDING, pool=POOLING, dropout=DROPOUT, num_classes=NUM_CLASSES)
#     log = model.training_model(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
#     logs[:, i] = log[:, 0]
#
#     """Prediction"""
#     predict = torch.argmax(model(test_x), dim=1)
#     prec_t = torch.eq(predict, test_y.T)
#     precision = torch.count_nonzero(prec_t) / prec_t.shape[1]
#     prec_l[i] = (precision.item())
#     print('Test Precision :', precision.item())
#
# print('Precisions:', prec_l)
# print('Error:', logs[-1, :])
# """Plot"""
# fig = plt.figure(figsize=(5, 3))
# for i in range(num_exps):
#     plt.plot(logs[:, i], label='K={}'.format(ks[i]))
#
# plt.legend(loc='upper right', ncol=3, fontsize=8)
# plt.xlabel('Epochs')
# plt.ylabel('CrossEntropyLoss')
# plt.yscale('log')
# plt.title('Experiment 3: Stride')
# plt.show()



"""
============ =====================================================================
Fine-tuning: POOLING
------------ ---------------------------------------------------------------------
"""

# num_exps = 6
# logs = np.zeros((EPOCHS, num_exps))
# prec_l = np.zeros(num_exps)
# ks = [2, 4, 8, 16, 32, 64]
# for i in range(num_exps):
#     pooling = ks[i]
#     """Training"""
#     model = ModelCNN(input_channel=INPUT_CHANNEL, input_len=INPUT_LENGTH, latent_channel=LATENT_CHANNEL, k=KERNEL_SIZE,
#                      s=STRIDE, p=PADDING, pool=pooling, dropout=DROPOUT, num_classes=NUM_CLASSES)
#     log = model.training_model(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
#     logs[:, i] = log[:, 0]
#
#     """Prediction"""
#     predict = torch.argmax(model(test_x), dim=1)
#     prec_t = torch.eq(predict, test_y.T)
#     precision = torch.count_nonzero(prec_t) / prec_t.shape[1]
#     prec_l[i] = (precision.item())
#     print('Test Precision :', precision.item())
#
# print('Precisions:', prec_l)
# print('Error:', logs[-1, :])
# """Plot"""
# fig = plt.figure(figsize=(5, 3))
# for i in range(num_exps):
#     plt.plot(logs[:, i], label='K={}'.format(ks[i]))
#
# plt.legend(loc='upper right', ncol=3, fontsize=8)
# plt.xlabel('Epochs')
# plt.ylabel('CrossEntropyLoss')
# plt.yscale('log')
# plt.title('Experiment 4: Pooling')
# plt.show()


"""
============ =====================================================================
Fine-tuning: DROPOUT
------------ ---------------------------------------------------------------------
"""

# num_exps = 4
# logs = np.zeros((EPOCHS, num_exps))
# prec_l = np.zeros(num_exps)
# ks = [0.25, 0.5, 0.75, 0.9]
# for i in range(num_exps):
#     dropout = ks[i]
#     """Training"""
#     model = ModelCNN(input_channel=INPUT_CHANNEL, input_len=INPUT_LENGTH, latent_channel=LATENT_CHANNEL, k=KERNEL_SIZE,
#                      s=STRIDE, p=PADDING, pool=POOLING, dropout=dropout, num_classes=NUM_CLASSES)
#     log = model.training_model(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
#     logs[:, i] = log[:, 0]
#
#     """Prediction"""
#     predict = torch.argmax(model(test_x), dim=1)
#     prec_t = torch.eq(predict, test_y.T)
#     precision = torch.count_nonzero(prec_t) / prec_t.shape[1]
#     prec_l[i] = (precision.item())
#     print('Test Precision :', precision.item())
#
# print('Precisions:', prec_l)
# print('Error:', logs[-1, :])
# """Plot"""
# fig = plt.figure(figsize=(5, 3))
# for i in range(num_exps):
#     plt.plot(logs[:, i], label='K={}'.format(ks[i]))
#
# plt.legend(loc='upper right', ncol=3, fontsize=8)
# plt.xlabel('Epochs')
# plt.ylabel('CrossEntropyLoss')
# plt.yscale('log')
# plt.title('Experiment 5: Dropout')
# plt.show()


"""
============ =====================================================================
Fine-tuning: LEARNING RATE
------------ ---------------------------------------------------------------------
"""

# num_exps = 4
# logs = np.zeros((EPOCHS, num_exps))
# prec_l = np.zeros(num_exps)
# ks = [0.01, 0.001, 0.0001, 0.00001]
# for i in range(num_exps):
#     lr = ks[i]
#     """Training"""
#     model = ModelCNN(input_channel=INPUT_CHANNEL, input_len=INPUT_LENGTH, latent_channel=LATENT_CHANNEL, k=KERNEL_SIZE,
#                      s=STRIDE, p=PADDING, pool=POOLING, dropout=DROPOUT, num_classes=NUM_CLASSES)
#     log = model.training_model(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=lr)
#     logs[:, i] = log[:, 0]
#
#     """Prediction"""
#     predict = torch.argmax(model(test_x), dim=1)
#     prec_t = torch.eq(predict, test_y.T)
#     precision = torch.count_nonzero(prec_t) / prec_t.shape[1]
#     prec_l[i] = (precision.item())
#     print('Test Precision :', precision.item())
#
# print('Precisions:', prec_l)
# print('Error:', logs[-1, :])
# """Plot"""
# fig = plt.figure(figsize=(5, 3))
# for i in range(num_exps):
#     plt.plot(logs[:, i], label='K={}'.format(ks[i]))
#
# plt.legend(loc='upper right', ncol=3, fontsize=8)
# plt.xlabel('Epochs')
# plt.ylabel('CrossEntropyLoss')
# plt.yscale('log')
# plt.title('Experiment 6: Learning Rate')
# plt.show()
