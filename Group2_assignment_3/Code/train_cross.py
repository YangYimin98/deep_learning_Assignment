# from datetime import datetime

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
KERNEL_SIZE = 7
STRIDE = 1
PADDING = 2
POOLING = 16
DROPOUT = 0.9
EPOCHS = 500
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
L1_WEIGHT = 0.0001

"""Training"""
m_intra_train = get_subject_data_set(p1='Cross', p2='train',
                                     subject_state=['rest', 'task_motor', 'task_story_math', 'task_working_memory'],
                                     subject_id=None, resampling_scale=16)
m_intra_test1 = get_subject_data_set(p1='Cross', p2='test1',
                                     subject_state=['rest', 'task_motor', 'task_story_math', 'task_working_memory'],
                                     subject_id=None, resampling_scale=16)
m_intra_test2 = get_subject_data_set(p1='Cross', p2='test2',
                                     subject_state=['rest', 'task_motor', 'task_story_math', 'task_working_memory'],
                                     subject_id=None, resampling_scale=16)
m_intra_test3 = get_subject_data_set(p1='Cross', p2='test3',
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
test_size1 = len(m_intra_test1)
test_x1 = np.zeros((test_size1, m_intra_test1[0][0].shape[0], m_intra_test1[0][0].shape[1]))
test_y1 = np.zeros((test_size1, 1))
for t_index in range(test_size1):
    test_x1[t_index, :, :] = m_intra_test1[t_index][0]
    test_y1[t_index, :] = m_intra_test1[t_index][1]

test_size2 = len(m_intra_test2)
test_x2 = np.zeros((test_size2, m_intra_test2[0][0].shape[0], m_intra_test2[0][0].shape[1]))
test_y2 = np.zeros((test_size2, 1))
for t_index in range(test_size2):
    test_x2[t_index, :, :] = m_intra_test2[t_index][0]
    test_y2[t_index, :] = m_intra_test2[t_index][1]

test_size3 = len(m_intra_test3)
test_x3 = np.zeros((test_size3, m_intra_test3[0][0].shape[0], m_intra_test3[0][0].shape[1]))
test_y3 = np.zeros((test_size3, 1))
for t_index in range(test_size3):
    test_x3[t_index, :, :] = m_intra_test3[t_index][0]
    test_y3[t_index, :] = m_intra_test3[t_index][1]

test_x = Variable(torch.Tensor(np.concatenate((test_x1, test_x2, test_x3), axis=0)))
test_y = Variable(torch.Tensor(np.concatenate((test_y1, test_y2, test_y3), axis=0))).long()
print('Testing data set x size: ', test_x.shape)
print('Testing data set y size: ', test_y.shape)

"""Training"""

# model = torch.load('intra_0750.pkl')

model = ModelCNN(input_channel=INPUT_CHANNEL, input_len=INPUT_LENGTH, latent_channel=LATENT_CHANNEL, k=KERNEL_SIZE,
                 s=STRIDE, p=PADDING, pool=POOLING, dropout=DROPOUT, num_classes=NUM_CLASSES)
log = model.training_model(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, l1_weight=L1_WEIGHT)

predict = torch.argmax(model(test_x), dim=1)
prec_t = torch.eq(predict, test_y.T)
precision = torch.count_nonzero(prec_t) / prec_t.shape[1]
print('Test Precision :', precision.item())
torch.save(model, 'Cross.pkl')


for param in model.parameters():
    param.requires_grad = False
# num_fc = model.linear2.in_features
# model.conv1 = nn.Conv1d(LATENT_CHANNEL, LATENT_CHANNEL, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)
model.linear2 = nn.Linear(np.int(INPUT_LENGTH / POOLING) * LATENT_CHANNEL, NUM_CLASSES)


"""Transfer learning"""

"""Training data set"""
m_intra_train = get_subject_data_set(p1='Intra', p2='train',
                                     subject_state=['rest', 'task_motor', 'task_story_math', 'task_working_memory'],
                                     subject_id=None, resampling_scale=16)
train_size_intra = len(m_intra_train)
num_batches_intra = np.int(train_size_intra / BATCH_SIZE)
train_x_intra = np.zeros((num_batches_intra, BATCH_SIZE, m_intra_train[0][0].shape[0], m_intra_train[0][0].shape[1]))
train_y_intra = np.zeros((num_batches_intra, BATCH_SIZE, 1))
for b_index in range(num_batches_intra):
    for s_index in range(BATCH_SIZE):
        train_x_intra[b_index, s_index, :, :] = m_intra_train[b_index * BATCH_SIZE + s_index][0]
        train_y_intra[b_index, s_index, :] = m_intra_train[b_index * BATCH_SIZE + s_index][1]
train_x_intra = Variable(torch.Tensor(train_x_intra))
train_y_intra = Variable(torch.Tensor(train_y_intra)).long()
print('Intra training data set x size: ', train_x_intra.shape)
print('Intra training data set y size: ', train_y_intra.shape)

log = model.training_model(train_x_intra, train_y_intra, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, l1_weight=L1_WEIGHT)


"""Prediction"""
predict = torch.argmax(model(test_x), dim=1)
prec_t = torch.eq(predict, test_y.T)
precision = torch.count_nonzero(prec_t) / prec_t.shape[1]
print('Test Precision :', precision.item())
torch.save(model, 'Cross_transferred.pkl')


