import gc
import os
import re
import torch
from torch.autograd import Variable
import h5py
import numpy as np
from scipy.signal import resample, filtfilt, butter

CLASS = {'rest': np.array([0.0]), 'task_motor': np.array([1.0]),
         'task_story_math': np.array([2.0]), 'task_working_memory': np.array([3.0])}


def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name


def get_data_set(file_path):
    with h5py.File(file_path, 'r') as f:
        dataset_name = get_dataset_name(file_path)
        matrix = f.get(dataset_name)[()]
        print('Data size: ', matrix.shape, ' from ', file_path)
    return matrix


def get_subject_data_set(p1, p2, subject_state, subject_id=None, resampling_scale=8):
    path_dir = '/'.join([p1, p2])
    file_list = os.listdir(path_dir)
    f_content = []

    for s in subject_state:
        for fp in file_list:
            if subject_id:
                p3 = '_'.join([s, subject_id])
            else:
                p3 = s

            if re.search(p3, fp):
                f_class = CLASS[s]
                m = get_data_set(path_dir + '/' + fp)

                # b, a = butter(8, [0.01, 0.5], 'bandpass')
                # m = filtfilt(b, a, m)
                m = resample(m, np.int(m.shape[1] / resampling_scale), axis=1)

                m_avg = np.mean(m, axis=1, keepdims=True)
                m_std = np.std(m, axis=1, keepdims=True)
                m = (m - m_avg) / m_std
                m = Variable(torch.Tensor(m))

                f_content.append((m.T, f_class, fp, m_avg, m_std))
            gc.collect()
    return f_content


# m_l = get_subject_data_set(p1='Cross', p2='train',
#                            subject_state=['rest', 'task_motor', 'task_story_math', 'task_working_memory'],
#                            subject_id=None)

# filename_path = "Intra/train/rest_105923_1.h5"
# m = get_data_set(filename_path)
