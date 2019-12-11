import configparser as ConfigParser
from optparse import OptionParser
import argparse
import numpy as np 
import math


def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError 

def read_conf(cfg_path, options):
    
    cfg_file=cfg_path
    print("read config file: ", cfg_file)
    Config = ConfigParser.ConfigParser()
    Config.read(cfg_file)

    #[windowing]
    options.windowing = argparse.Namespace()
    options.windowing.fs = int(Config.get('windowing', 'fs'))
    options.windowing.cw_len = int(Config.get('windowing', 'cw_len'))
    options.windowing.cw_shift = int(Config.get('windowing', 'cw_shift'))

    #[cnn]
    options.cnn = argparse.Namespace()
    options.cnn.cnn_input_dim = int(Config.get('cnn', 'cnn_input_dim'))
    options.cnn.cnn_N_filt = list(map(int, Config.get('cnn', 'cnn_N_filt').split(',')))
    options.cnn.cnn_len_filt = list(map(int, Config.get('cnn', 'cnn_len_filt').split(',')))
    options.cnn.cnn_max_pool_len = list(map(int, Config.get('cnn', 'cnn_max_pool_len').split(',')))
    options.cnn.cnn_use_laynorm_inp = str_to_bool(Config.get('cnn', 'cnn_use_laynorm_inp'))
    options.cnn.cnn_use_batchnorm_inp = str_to_bool(Config.get('cnn', 'cnn_use_batchnorm_inp'))
    options.cnn.cnn_use_laynorm = list(map(str_to_bool, Config.get('cnn', 'cnn_use_laynorm').split(',')))
    options.cnn.cnn_use_batchnorm = list(map(str_to_bool, Config.get('cnn', 'cnn_use_batchnorm').split(',')))
    options.cnn.cnn_act = list(map(str, Config.get('cnn', 'cnn_act').split(',')))
    options.cnn.cnn_drop = list(map(float, Config.get('cnn', 'cnn_drop').split(',')))
    options.cnn.arch_lr = float(Config.get('cnn', 'arch_lr'))
    options.cnn.arch_opt = str(Config.get('cnn', 'arch_opt'))
    options.cnn.arch_opt_alpha = float(Config.get('cnn', 'arch_opt_alpha'))
    options.cnn.lr_decay_step = int(Config.get('cnn', 'lr_decay_step'))
    options.cnn.lr_decay_factor = float(Config.get('cnn', 'lr_decay_factor'))

    #[dnn]
    options.dnn = argparse.Namespace()
    options.dnn.fc_input_dim = int(Config.get('dnn', 'fc_input_dim'))
    options.dnn.fc_lay = list(map(int, Config.get('dnn', 'fc_lay').split(',')))
    options.dnn.fc_drop = list(map(float, Config.get('dnn', 'fc_drop').split(',')))
    options.dnn.fc_use_laynorm_inp = str_to_bool(Config.get('dnn', 'fc_use_laynorm_inp'))
    options.dnn.fc_use_batchnorm_inp = str_to_bool(Config.get('dnn', 'fc_use_batchnorm_inp'))
    options.dnn.fc_use_batchnorm = list(map(str_to_bool, Config.get('dnn', 'fc_use_batchnorm').split(',')))
    options.dnn.fc_use_laynorm = list(map(str_to_bool, Config.get('dnn', 'fc_use_laynorm').split(',')))
    options.dnn.fc_act = list(map(str, Config.get('dnn', 'fc_act').split(',')))
    options.dnn.arch_lr = float(Config.get('dnn', 'arch_lr'))
    options.dnn.arch_opt = str(Config.get('dnn', 'arch_opt'))
    options.dnn.lr_decay_step = int(Config.get('dnn', 'lr_decay_step'))
    options.dnn.lr_decay_factor = float(Config.get('dnn', 'lr_decay_factor'))

    #[class]
    options.classifier = argparse.Namespace()
    options.classifier.fc_input_dim = int(Config.get('classifier', 'fc_input_dim'))
    options.classifier.fc_lay = list(map(int, Config.get('classifier', 'fc_lay').split(',')))
    options.classifier.fc_drop = list(map(float, Config.get('classifier', 'fc_drop').split(',')))
    options.classifier.fc_use_laynorm_inp = str_to_bool(Config.get('classifier', 'fc_use_laynorm_inp'))
    options.classifier.fc_use_batchnorm_inp = str_to_bool(Config.get('classifier', 'fc_use_batchnorm_inp'))
    options.classifier.fc_use_batchnorm = list(map(str_to_bool, Config.get('classifier', 'fc_use_batchnorm').split(',')))
    options.classifier.fc_use_laynorm = list(map(str_to_bool, Config.get('classifier', 'fc_use_laynorm').split(',')))
    options.classifier.fc_act = list(map(str, Config.get('classifier', 'fc_act').split(',')))
    options.classifier.arch_lr = float(Config.get('classifier', 'arch_lr'))
    options.classifier.arch_opt = str(Config.get('classifier', 'arch_opt'))
    options.classifier.lr_decay_step = int(Config.get('classifier', 'lr_decay_step'))
    options.classifier.lr_decay_factor = float(Config.get('classifier', 'lr_decay_factor'))

    #[optimization]
    options.optimization = argparse.Namespace()
    options.optimization.lr=float(Config.get('optimization', 'lr'))
    options.optimization.batch_size=int(Config.get('optimization', 'batch_size'))
    options.optimization.N_epochs=int(Config.get('optimization', 'N_epochs'))
    options.optimization.N_eval_epoch=int(Config.get('optimization', 'N_eval_epoch'))
    options.optimization.print_every=int(Config.get('optimization', 'print_every'))
    options.optimization.seed=int(Config.get('optimization', 'seed'))
    
    return options


def get_dict_from_args(keys, args):
    data = {}
    for key in keys:
        data[key] = getattr(args, key)
    return data


def SNR(pred:np.array, label:np.array):
     assert pred.shape == label.shape, "the shape of pred and label must be the same"
     pred, label = (pred+1)/2, (label+1)/2
     if len(pred.shape)>1:
         sigma_s_square = np.mean(label*label, axis=1)
         sigma_e_square = np.mean((pred-label)*(pred-label), axis=1)
         snr = 10*np.log10((sigma_s_square/max(sigma_e_square, 1e-9)))
         snr = snr.mean()
     else:
         sigma_s_square = np.mean(label*label)
         # print('sigma_s_square:', sigma_s_square)
         sigma_e_square = np.mean((pred-label)*(pred-label))
         # print('sigma_e_square:', sigma_e_square)
         # print(sigma_s_square/max(sigma_e_square, 1e-9))
         snr = 10*np.log10((sigma_s_square/max(sigma_e_square, 1e-9)))
     return snr

from pypesq import pesq
def PESQ(ref, rec, sr=16000):
    return pesq(ref, rec, sr)


def MOS_LQO(_pesq):
    x = _pesq
    y = 0.999+4/(1+math.exp(-1.4945*x+4.6607))
    return y

import math
def get_noise_scale(SNR):
    return math.pow(10, SNR/10.0)