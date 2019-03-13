from __future__ import division, print_function, unicode_literals
from sacred import Ingredient, Experiment
from torchvision import datasets, transforms
import torch 
import numpy as np
import pandas as pd
import peak_detection as peak_det


data_ingredient = Ingredient('dataset')

@data_ingredient.config
def cfg():
    name = 'data.csv'  # dataset filename
    source = '../data/'
    shuffle = True #only for training set by default

# x : (N, win_len)
# minPeakDist = 5, minPeakHeight = None, maxPeakHeight = None, \
#  minPeakTh = None, maxPeakTh = None, height_type = None, remove_trend = False
def detect_changes(x, **kwargs):
    def is_changed(window, **kwargs):
        window = -window
        obj = peak_det.peak_detection(window)
        peak_locs_pos = obj.findPeaks(**kwargs)
        if len(peak_locs_pos) > 1:
            max_val = window[peak_locs_pos]
            idx_loc_max = np.argmax(max_val)
            peak_loc_pos = peak_locs_pos[idx_loc_max]
            height = obj.height[idx_loc_max]
        elif len(peak_locs_pos)==0:
            return 0
        else:
            peak_loc_pos = peak_locs_pos[0]
            height = obj.height[0]
#        if 'delta' in kwargs:
#            delta = kwargs['delta']
#        else:
#            delta = 0.3
        left_part = window[:peak_loc_pos + 1]
        left_part_diff_avg = left_part[-1] - left_part[0]
        right_part = window[peak_loc_pos:]
        right_part_diff_avg = right_part[0] - right_part[-1]
        # equal signs
        if right_part_diff_avg * left_part_diff_avg > 0:
            return 1
        return 0
    labels = np.apply_along_axis(lambda z: is_changed(z, **kwargs), 2, x)
    min_values = np.apply_along_axis(lambda z: z.min())
    return labels, min_values

def get_sequences(x, win_past_len, win_pred_len, stride):
    windows = rolling_window(x, win_past_len + win_pred_len, stride=stride)
    win_past = windows[:, :win_past_len, :]
    win_pred = windows[:, win_past_len:, :]
    return win_past, win_pred

def basic_tasks(x, win_past_len, win_pred_len, stride):
    win_past, win_pred = get_sequences(x, win_past_len, win_pred_len, stride)
    labels, min_values = detect_changes(win_pred, minPeakDist=25, minPeakHeight=.05, maxPeakHeight=None, \
        minPeakTh=None, maxPeakTh=None, remove_trend=False)
    return win_past, win_pred[...,2:6], labels, min_values
    
def couples_task(windows):
    '''
    (x(t), u(t)) vs (x(t), u(t_rand))
    '''
    inputs = windows[:,:,2:6]
    cgm = windows[:,:,1]
    rand_idx = np.random.randint(low=0, high=inputs.shape[0], size=inputs.shape[0] // 2)
    inputs[:inputs.shape[0] // 2] = inputs[rand_idx]
    labels = np.ones(inputs.shape[0]) # random
    labels[:inputs.shape[0] // 2] = 0 # not random
    return inputs, cgm, labels

def future_task(a, stride, lag):
    '''
    Get rolled windows and get future window with a lag
    lag: units of strides
    '''
    future_win = a[int(stride * lag):]
    half_len = future_win.shape[0] // 2
    rand_idx = np.random.randint(low=0, high=future_win.shape[0], size=half_len)
    future_win[:half_len] = future_win[future_win]
    labels = np.ones(half_len)
    labels[:half_len] = 0
    return a, future_win, labels

def rolling_window(a, window, stride=1):
    a = np.array(a)
    if len(a.shape) > 1:
        # a: N x p
        arrays = [_rolling_window(a[:,i], window, stride) for i in range(a.shape[1])]
        return np.stack(arrays, axis=1)
    else:
        return _rolling_window(a, window, stride)

def _rolling_window(a, window, stride=1):
    a = np.array(a).copy()
    #n_bytes = int(a.nbytes / len(a))
    shape = a.shape[1:] + (int((a.shape[0] - window) /stride) + 1, window)
    strides = list(a.strides)
    strides[0] *= stride
    strides = tuple(strides) + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def get_dataset(opt, tasks=['basic'], shuffle=True):
    win_past_len = opt['win_past_len']
    stride = opt['stride']
    loaders = dict()
    if 'basic' in tasks:
        loaders['basic'] = []
        win_past, inp_win_pred, labels, min_values = basic_tasks(x, win_past_len, win_pred_len, stride)
        #data = np.stack(win_past, inp_win_pred)
        tds_cl = tnt.dataset.TensorDataset([win_past, inp_win_pred, labels])
        tds_reg = tnt.dataset.TensorDataset([win_past, inp_win_pred, min_values])
        loader_cl = th.utils.data.DataLoader(tds_cl, 
                        batch_size=opt['b'], shuffle=shuffle, 
                        num_workers=opt['j'], pin_memory=True)
        loaders['basic'].append(loader_cl)
        loader_reg = th.utils.data.DataLoader(tds_reg, 
                        batch_size=opt['b'], shuffle=shuffle, 
                        num_workers=opt['j'], pin_memory=True)
        loaders['basic'].append(loader_reg)
    return loaders


@data_ingredient.capture
def load_data(ctx, name, source, shuffle):
    opt = ctx.opt
    if opt['seed']:
        np.random.seed(opt['seed'])
    
    data = pd.read_csv('../data.csv')
    columns = {'time', 'CGM', 'meal', 'basal', 'bolus',
       'insulin_basal_value', 'fault_basal', 'fault_bolus', 'ID'}
    # if opt['personalized']:
    #     cols.append('CR')

    data = data[columns]
    for col in columns:
        if 'float' in str(data[col].dtype) or col == 'CGM':
            data[col] = data[col].astype(np.float32)

    # shuffle by random IDs
    IDperm = np.random.permutation(data.ID.unique())
    ctx.IDPerm = IDperm
    tr_pat = 60
    seq_len = len(data[data.ID == 1].CGM)
    train_data = data[data.ID.isin(IDperm[:tr_pat])].pivot(index='ID',
                                                columns='time',
                                                values=columns-{'time', 
                                                'ID'}).to_numpy().reshape((tr_pat, -1, seq_len)).transpose(0,2,1)
    test_data = data[data.ID.isin(IDperm[:tr_pat])].pivot(index='ID',
                                                columns='time',
                                                values=columns-{'time', 
                                                'ID'}).to_numpy().reshape((tr_pat, -1, seq_len)).transpose(0,2,1)

    train_loaders = get_dataset(opt, train_data[:,:4000,:])
    test_loaders = get_dataset(opt, test_data[:,:4000,:])

    return train_loaders, test_loaders