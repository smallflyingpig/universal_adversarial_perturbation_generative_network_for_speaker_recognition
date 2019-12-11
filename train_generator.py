import os
import tqdm
import argparse
import numpy as np
import pandas as pd 
from copy import deepcopy
import random
import logging
import math
from functools import partial
from tensorboardX import SummaryWriter

from common.trainer import ClassifierTrainer, load_checkpoint, save_checkpoint
from common.dataset import TIMIT_speaker_norm, LibriSpeech_speaker
from common.utils import read_conf, get_dict_from_args
from common.model import SincClassifier

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader

from generator import Generator1D


@torch.no_grad()
def evaluate(model, test_dataset, cost, target=-1, noise_scale=1):
    test_dataloader = DataLoader(test_dataset, 128, shuffle=False, num_workers=8, pin_memory=True)
    model.eval()
    bar = tqdm.tqdm(test_dataloader)
    loss_all = {}
    err_fake_all = {}
    noise_all = {}
    for idx, data in enumerate(bar):
        wav_data, speaker_id, norm_factor = data
        batch_size = wav_data.shape[0]
        noise_dim = model.noise_dim
        noise = torch.randn(size=(batch_size, noise_dim))
        noise, wav_data, speaker_id = noise.float().cuda(), wav_data.float().cuda(), speaker_id.long().cuda()
        norm_factor = norm_factor.unsqueeze(1).repeat(1, wav_data.shape[1]).float().cuda()

        loss_func = cost
        with torch.no_grad():
            pout = model.forward(noise)
            print(pout.shape)
            pout = (pout*noise_scale + wav_data*norm_factor).clamp_(-1,1)/norm_factor
            labels = {"speaker":speaker_id, "norm":wav_data}
            loss_total, loss_dict, loss_dict_grad, pred_dict, label_dict = loss_func(pout, labels)
        pred = torch.max(pred_dict['speaker'], dim=1)[1]
        if target<0:
            label = label_dict['speaker']
            err_speaker = torch.mean((pred != label).float()).detach().cpu().item()
        else:
            err_speaker = torch.mean((pred != target).float()).detach().cpu().item()
        
        err_dict = {"err_spk":err_speaker}
        err_str = get_dict_str(err_dict)

        loss_total = loss_total.detach().cpu().item()
        loss_str = get_dict_str(loss_dict)

        loss_dict.update(err_dict)
        noise = (pout.detach()-wav_data.detach())
        noise_mean, noise_std, noise_abs = torch.mean(noise).item(), torch.std(noise).item(), torch.mean(torch.abs(noise)).item()
        noise_dict = {"mean":noise_mean*1e3, "std":noise_std*1e3, "m_abs":noise_abs*1e3}
        noise_str = get_dict_str(noise_dict)
        loss_dict.update(noise_dict)
        def accumulate_dict(total_dict, item_dict, factor):
            for k,v in item_dict.items():
                total_dict[k] = total_dict.get(k,0)+v*factor
            return total_dict
        loss_all = accumulate_dict(loss_all, loss_dict, len(speaker_id))
        err_fake_all = accumulate_dict(err_fake_all, err_dict, len(speaker_id))
        noise_all = accumulate_dict(noise_all, noise_dict, len(speaker_id))
        
        bar.set_description("err:({}), noise(e-3):({}), batch size:{}".format(err_str, noise_str, len(speaker_id)))
    bar.close()
    def multiply_dict(data_dict, factor):
        for k,v in data_dict.items():
            data_dict[k] = v*factor
        return data_dict
    loss_all = multiply_dict(loss_all, 1.0/len(test_dataset))
    err_fake_all = multiply_dict(err_fake_all, 1.0/len(test_dataset))
    noise_all = multiply_dict(noise_all, 1.0/len(test_dataset))
    print(get_dict_str(loss_all), get_dict_str(err_fake_all), get_dict_str(noise_all))


@torch.no_grad()
def sentence_test(speaker_model, wav_data, wlen=3200, wshift=10, batch_size=128):
    """
    wav_data: B, L
    """
    wav_data = wav_data.squeeze()
    L = wav_data.shape[0]
    pred_all = []
    begin_idx = 0
    batch_data = []
    while begin_idx<L-wlen:
        batch_data.append(wav_data[begin_idx:begin_idx+wlen])
        if len(batch_data)>=batch_size:
            pred_batch = speaker_model(torch.stack(batch_data))
            pred_all.append(pred_batch)
            batch_data = []
        begin_idx += wshift
    if len(batch_data)>0:
        pred_batch = speaker_model(torch.stack(batch_data))
        pred_all.append(pred_batch)
    [val,best_class]=torch.max(torch.sum(torch.cat(pred_all, dim=0),dim=0),0)
    return best_class.detach().cpu().item()

import soundfile as sf
from common.utils import SNR, PESQ
from common.trainer import RunningAverage
@torch.no_grad()
def test_wav(model:Generator1D, filename_list, data_folder, out_folder, speaker_model=None, label_dict=None, target=-1, noise_scale=1):
    model.eval()
    if speaker_model: speaker_model.eval()
    noise_dim = model.noise_dim
    batch_size = 1
    bar = tqdm.tqdm(filename_list)
    averager = RunningAverage()
    pertutations = []
    pred_results = []
    save_every = 2000
    save_idx = 0
    for idx, filename in enumerate(bar):
        noise = torch.randn(size=(batch_size, noise_dim))
        real_data, fs = sf.read(os.path.join(data_folder, filename))
        real_data_norm, real_norm_factor = TIMIT_speaker_norm.preprocess(real_data)
        pout = model.forward(noise.float().cuda()).squeeze().detach().cpu().numpy()
        # print(np.abs(pout).mean())
        # cycle
        noise_all = np.concatenate([pout]*int(math.ceil(len(real_data)/float(len(pout)))))[:len(real_data)]
        fake_data = (noise_all*noise_scale + real_data).clip(-1,1)
        fake_data_norm = fake_data/np.abs(fake_data).max()
        # save data
        output_filename = os.path.join(out_folder, filename)
        if not os.path.exists(os.path.dirname(output_filename)): 
            os.makedirs(os.path.dirname(output_filename))
        # print(fake_data.shape)
        sf.write(output_filename, fake_data, fs)
        snr = SNR(fake_data, real_data)
        pesq = PESQ(real_data, fake_data, fs)
        averager.update({"SNR":snr, "PESQ":pesq}, {"SNR":snr, "PESQ":pesq})
        output_str = "SNR:{:5.2f}, PESQ:{:5.2f}".format(snr, pesq)
        pertutations.append((real_data-fake_data).astype(np.float16))
        if speaker_model:
            label = label_dict[filename]
            pred_fake = sentence_test(speaker_model, torch.from_numpy(fake_data_norm).float().cuda().unsqueeze(0))
            if target != -1:
                err_rate = (pred_fake == target)
                averager.update({"err_rate":err_rate}, {"err_rate":1})
                pred_real = sentence_test(speaker_model, torch.from_numpy(real_data_norm).float().cuda().unsqueeze(0))
                averager.update({"err_rate_raw":pred_real!=label, "target_rate_raw":pred_real==target}, {"err_rate_raw":1, "target_rate_raw":1})
                pred_results.append({'file':filename, 'pred_real':pred_real, 'pred_fake':pred_fake, 'label':label})
            else:
                err_rate = (pred_fake != label)
                averager.update({"err_rate":err_rate}, {"err_rate":1})
                pred_results.append({'file':filename, 'pred_fake':pred_fake, 'label':label})
            output_str += ", real/fake:{}/{}, data len:{}".format(label, pred_fake, fake_data.shape)
        bar.set_description(output_str+filename)
        if len(pertutations)>=save_every:
            np.save(os.path.join(out_folder, "pertutation.{}.npy".format(save_idx)), (pertutations))
            pertutations = []
            if len(pred_results)>0:
                pd.DataFrame(pred_results).to_csv(os.path.join(out_folder, "pred_results.{}.csv".format(save_idx)))
                pred_results = []
            save_idx += 1

    np.save(os.path.join(out_folder, "pertutation.{}.npy".format(save_idx)), (pertutations))
    if len(pred_results)>0:
        pd.DataFrame(pred_results).to_csv(os.path.join(out_folder, "pred_results.{}.csv".format(save_idx)))
    bar.close()
    avg = averager.average()
    print(get_dict_str(avg))


def test_interpolation(model:Generator1D, filename_list, data_folder, out_folder, speaker_model=None, label_dict=None, target=-1, noise_scale=1, beta=0):
    model.eval()
    if speaker_model: speaker_model.eval()
    noise_dim = model.noise_dim
    batch_size = 1
    bar = tqdm.tqdm(filename_list)
    averager = RunningAverage()
    pertutations = []
    pred_results = []
    save_every = 2000
    save_idx = 0
    noise1 = torch.randn(size=(batch_size, noise_dim))
    noise2 = torch.randn(size=(batch_size, noise_dim))
    noise = noise1 * (1-beta) + noise2 * beta
    pout = model.forward(noise.float().cuda()).squeeze().detach().cpu().numpy()
    if beta < 0:
        pout = torch.randn(size=pout.shape).mul_(-beta).numpy()
    for idx, filename in enumerate(bar):
        real_data, fs = sf.read(os.path.join(data_folder, filename))
        real_data_norm, real_norm_factor = TIMIT_speaker_norm.preprocess(real_data)
        # print(np.abs(pout).mean())
        # cycle
        if beta < 0:
            noise_all = torch.randn(real_data.shape).mul_(-beta).numpy()
        else:
            noise_all = np.concatenate([pout]*int(math.ceil(len(real_data)/float(len(pout)))))[:len(real_data)]
        fake_data = (noise_all*noise_scale + real_data).clip(-1,1)
        fake_data_norm = fake_data/np.abs(fake_data).max()
        # save data
        output_filename = os.path.join(out_folder, filename)
        if not os.path.exists(os.path.dirname(output_filename)): 
            os.makedirs(os.path.dirname(output_filename))
        # print(fake_data.shape)
        sf.write(output_filename, fake_data, fs)
        snr = SNR(fake_data, real_data)
        pesq = PESQ(real_data, fake_data, fs)
        averager.update({"SNR":snr, "PESQ":pesq}, {"SNR":snr, "PESQ":pesq})
        output_str = "SNR:{:5.2f}, PESQ:{:5.2f}".format(snr, pesq)
        pertutations.append((real_data-fake_data).astype(np.float16))
        if speaker_model:
            label = label_dict[filename]
            pred_fake = sentence_test(speaker_model, torch.from_numpy(fake_data_norm).float().cuda().unsqueeze(0))
            if target != -1:
                err_rate = (pred_fake == target)
                averager.update({"err_rate":err_rate}, {"err_rate":1})
                pred_real = sentence_test(speaker_model, torch.from_numpy(real_data_norm).float().cuda().unsqueeze(0))
                averager.update({"err_rate_raw":pred_real!=label, "target_rate_raw":pred_real==target}, {"err_rate_raw":1, "target_rate_raw":1})
                pred_results.append({'file':filename, 'pred_real':pred_real, 'pred_fake':pred_fake, 'label':label})
            else:
                err_rate = (pred_fake != label)
                averager.update({"err_rate":err_rate}, {"err_rate":1})
                pred_results.append({'file':filename, 'pred_fake':pred_fake, 'label':label})
            output_str += ", real/fake:{}/{}, data len:{}".format(label, pred_fake, fake_data.shape)
        bar.set_description(output_str+filename)
        if len(pertutations)>=save_every:
            np.save(os.path.join(out_folder, "pertutation.{}.npy".format(save_idx)), (pertutations))
            pertutations = []
            if len(pred_results)>0:
                pd.DataFrame(pred_results).to_csv(os.path.join(out_folder, "pred_results.{}.csv".format(save_idx)))
                pred_results = []
            save_idx += 1

    np.save(os.path.join(out_folder, "pertutation.{}.npy".format(save_idx)), (pertutations))
    if len(pred_results)>0:
        pd.DataFrame(pred_results).to_csv(os.path.join(out_folder, "pred_results.{}.csv".format(save_idx)))
    bar.close()
    avg = averager.average()
    print(get_dict_str(avg))


def get_dict_str(d):
    s = ','.join(["{}:{:5.3f}".format(k,v) for k,v in d.items()])
    return s

grads = {}
def save_grad(v):
    def hook(grad):
        grads[v] = grad
    return hook

def batch_process_generator(model:Generator1D, data, train_mode=True, **kwargs):
    wav_data, speaker_id, norm_factor = data
    batch_size = wav_data.shape[0]
    noise_dim = model.noise_dim
    noise = torch.randn(size=(batch_size, noise_dim))
    noise, wav_data, speaker_id = noise.float().cuda(), wav_data.float().cuda(), speaker_id.long().cuda()
    noise_scale = kwargs.get("noise_scale", 1)
    target = kwargs.get("target", -1)
    norm_factor = norm_factor.unsqueeze(1).repeat(1, wav_data.shape[1]).float().cuda()
    if train_mode:
        model.train()
        optimizer, loss_func = kwargs.get("optimizer"), kwargs.get("loss_func")
        pout = model.forward(noise)
        pout.register_hook(save_grad('wav_data'))
        pout = (pout*noise_scale + wav_data*norm_factor).clamp_(-1,1)/norm_factor
        labels = {"speaker": speaker_id, "norm":wav_data}
        loss_total, loss_dict, loss_dict_grad, pred_dict, label_dict = loss_func(pout, labels)
        grad_dict = {}
        for k, l in loss_dict_grad.items():
            model.zero_grad()
            l.backward(retain_graph=True)
            grad_dict[k] = grads['wav_data'].abs().mean()
        model.zero_grad()
        loss_total.backward()
        grad_dict['total'] = grads['wav_data'].abs().mean()
        optimizer.step()

        pred = torch.max(pred_dict['speaker'], dim=1)[1]
        if target<0:
            label = label_dict['speaker']
            err_speaker = torch.mean((pred != label).float()).detach().cpu().item()
        else:
            err_speaker = torch.mean((pred == target).float()).detach().cpu().item()
        
        err_dict = {"err_spk":err_speaker}
        err_str = get_dict_str(err_dict)

        loss_total = loss_total.detach().item()
        loss_str = get_dict_str(loss_dict)

        loss_dict.update(err_dict)
        noise = (pout.detach()-wav_data.detach())
        noise_mean, noise_std, noise_abs = torch.mean(noise).item(), torch.std(noise).item(), torch.mean(torch.abs(noise)).item()
        noise_dict = {"mean":noise_mean*1e3, "std":noise_std*1e3, "m_abs":noise_abs*1e3}
        noise_str = get_dict_str(noise_dict)
        loss_dict.update(noise_dict)
        grad_dict = {k:v*1e3 for k,v in grad_dict.items()}
        grad_str = get_dict_str(grad_dict)
        loss_dict.update(grad_dict)
        rtn = {
            "output":"loss_total:{:6.3f}({}), err:({}), lr(e-3):[{:6.3f}], grad(e-3):({}), noise(e-3):({})".format(
                 loss_total, loss_str, err_str, optimizer.param_groups[0]['lr']*1e3, grad_str, noise_str),
            "vars":loss_dict,
            "count":{k:len(speaker_id) for k in loss_dict}
        }
    else: #eval
        model.eval()
        optimizer, loss_func = kwargs.get('optimizer'), kwargs.get('loss_func')
        with torch.no_grad():
            pout = model.forward(noise)
            pout = (pout*noise_scale + wav_data*norm_factor).clamp_(-1,1)/norm_factor
            labels = {"speaker":speaker_id, "norm":wav_data}
            loss_total, loss_dict, loss_dict_grad, pred_dict, label_dict = loss_func(pout, labels)
        pred = torch.max(pred_dict['speaker'], dim=1)[1]
        if target<0:
            label = label_dict['speaker']
            err_speaker = torch.mean((pred != label).float()).detach().cpu().item()
        else:
            err_speaker = torch.mean((pred == target).float()).detach().cpu().item()
        
        err_dict = {"err_spk":err_speaker}
        err_str = get_dict_str(err_dict)

        loss_total = loss_total.detach().cpu().item()
        loss_str = get_dict_str(loss_dict)

        loss_dict.update(err_dict)
        noise = (pout.detach()-wav_data.detach())
        noise_mean, noise_std, noise_abs = torch.mean(noise).item(), torch.std(noise).item(), torch.mean(torch.abs(noise)).item()
        noise_dict = {"mean":noise_mean*1e3, "std":noise_std*1e3, "m_abs":noise_abs*1e3}
        noise_str = get_dict_str(noise_dict)
        loss_dict.update(noise_dict)
        rtn = {
            "output":"loss:{:6.3f}({}), err:({}), noise(e-3):({})".format(
                 loss_total, loss_str, err_str, noise_str),
            "vars":loss_dict,
            "count":{k:len(speaker_id) for k in loss_dict}
        }
    return rtn



class EvalHook(object):
    def __init__(self):
        self.best_accu = 0
    
    def __call__(self, model:nn.Module, epoch_idx, output_dir, 
        eval_rtn:dict, test_rtn:dict, logger:logging.Logger, writer:SummaryWriter):
        # save model
        acc = eval_rtn.get('err_spk', 0)-eval_rtn.get('err_sph', 1)
        is_best = acc > self.best_accu
        self.best_accu = acc if is_best else self.best_accu
        model_filename = "epoch_{}.pth".format(epoch_idx)
        save_checkpoint(model, os.path.join(output_dir, model_filename), 
            meta={'epoch':epoch_idx})
        os.system(
            "ln -sf {} {}".format(os.path.abspath(os.path.join(output_dir, model_filename)), 
            os.path.join(output_dir, "latest.pth"))
            )
        if is_best:
            os.system(
            "ln -sf {} {}".format(os.path.abspath(os.path.join(output_dir, model_filename)), 
            os.path.join(output_dir, "best.pth"))
            )

        if logger is not None:
            logger.info("EvalHook: best accu: {:.3f}, is_best: {}".format(self.best_accu, is_best))


class UniversalLoss(object):
    def __init__(self, loss_all):
        self.loss_all = loss_all
    
    def loss_schedule(self):
        for k in self.loss_all.keys():
            self.loss_all[k]['factor'] += self.loss_all[k]['factor_grow']

    def __call__(self, pred, labels):
        loss_dict_grad = {}
        loss_dict = {}
        pred_dict = {}
        label_dict = {}
        for key, loss in self.loss_all.items():
            B = len(labels[key])
            model = loss.get('model', None)
            if model is not None:
                pred_shape = pred.shape
                repeat = pred_shape[1]//3200
                pred_this = pred.view(pred_shape[0]*repeat, pred_shape[1]//repeat)
                label_this = torch.stack([labels[key]]*repeat, dim=1).view(B*repeat)
                pred_this = model(pred_this)
            else:
                pred_this = pred
                label_this = labels[key]

            label = labels[key]
            loss_func = loss["loss_func"]
            loss_this = loss_func(pred, label) * loss['factor']
            loss_dict[key] = loss_this.detach().cpu().item()
            loss_dict_grad[key] = loss_this
            pred_dict[key] = pred_this.detach()
            label_dict[key] = label_this.detach()
        loss_list = [v for k,v in loss_dict_grad.items()]
        loss_total = sum(loss_list)
        # loss_total = loss_dict_grad['norm'] * self.loss_all['norm']['factor']
        loss_dict["loss_total"] = loss_total.detach().cpu().item()
        return loss_total, loss_dict, loss_dict_grad, pred_dict, label_dict
            


class SpeakerLoss(object):
    def __init__(self, model:SincClassifier, clamp=1, mul=0.01, margin=100):
        self.model = model
        self.model.eval()
        self.clamp = clamp
        self.mul = mul
        self.margin = margin

    def __call__(self, pred, label):
        B = len(label)
        pred_shape = pred.shape
        repeat = pred.shape[1]//3200
        pred = pred.view(pred_shape[0]*repeat, pred_shape[1]//repeat)
        label = torch.stack([label]*repeat, dim=1).view(B*repeat)
        B = len(label)

        # pred = F.softmax(self.model(pred), dim=1)
        pred = self.model(pred)

        max_data, max_idx = torch.topk(pred, k=2, dim=1)
        pred_true = max_idx[:,0]==label
        pred_false = max_idx[:, 0] != label

        loss_true = pred[torch.arange(B), label][pred_true]-pred[torch.arange(B), max_idx[:, 1]][pred_true]+self.margin
        loss_true = torch.sum(loss_true.mul(self.mul))/(len(loss_true)+1e-5)

        loss_false = (pred[torch.arange(B), label][pred_false]-pred[torch.arange(B), max_idx[:,0]][pred_false]+self.margin)
        loss_false = loss_false[loss_false>0]
        loss_false = torch.sum(loss_false.mul(self.mul))/(len(loss_false)+1e-5)
        # print(select.sum())
        # loss = torch.sum(loss_true.mul(self.mul).clamp_(-self.clamp, self.clamp))/(len(loss_true)+1e-5) + \
        #     torch.sum(loss_false.mul(self.mul).clamp_(-self.clamp, self.clamp))/(len(loss_false)+1e-5)
        loss = loss_true + loss_false
        return loss



class SpeakerLossTarget(object):
    def __init__(self, model:SincClassifier, target=0, clamp=100, mul=0.01, margin=100):
        self.model = model
        self.target = int(target)
        self.model.eval()
        self.clamp = clamp
        self.mul = mul
        self.margin = margin

    def __call__(self, pred, label):
        B = len(label)
        pred_shape = pred.shape
        repeat = pred.shape[1]//3200
        pred = pred.view(pred_shape[0]*repeat, pred_shape[1]//repeat)
        label = torch.stack([label]*repeat, dim=1).view(B*repeat)
        B = len(label)

        pred = self.model(pred)

        max_data, max_idx = torch.topk(pred, k=2, dim=1)
        # print(max_idx)

        pred_true = max_idx[:,0]==self.target
        pred_false = max_idx[:, 0] != self.target
        # print(pred_true, pred_true.sum(), pred_false, pred_false.sum())
        loss_list = []
        if pred_true.sum()>0:
            loss_true = pred[torch.arange(B), self.target][pred_true]-pred[torch.arange(B), max_idx[:, 1]][pred_true]
            loss_true = loss_true[loss_true<self.margin]
            # loss_true = torch.sum(loss_true.mul(self.mul).clamp_(-self.clamp, self.clamp))/(len(loss_true)+1e-5)
            loss_true = torch.sum(loss_true.mul(self.mul))/(len(loss_true)+1e-5)
            # print("loss true:", loss_true, pred_true.sum())
            loss_list.append(loss_true)
        if pred_false.sum()>0:
            loss_false = (pred[torch.arange(B), self.target][pred_false]-pred[torch.arange(B), max_idx[:,0]][pred_false])
            loss_false = torch.sum(loss_false.mul(self.mul))/(len(loss_false)+1e-5)
            # print("loss false:", loss_false, pred_false.sum())
            loss_list.append(loss_false)
        # print(loss_true.detach().cpu().mean().numpy(), loss_false.detach().cpu().mean().numpy())
        if len(loss_list)>0:
            loss = -sum(loss_list)
        else:
            raise ValueError
        # loss = -(torch.sum(loss_true.mul(self.mul).clamp_(-self.clamp, self.clamp))/(len(loss_true)+1e-5) + \
        #     torch.sum(loss_false.mul(self.mul).clamp_(-self.clamp, self.clamp))/(len(loss_false)+1e-5))
        return loss


class MSEWithThreshold(object):
    def __init__(self, threshold=0.05, order=2):
        norm = {1:self.l1_with_threshold, 2:self.l2_with_threshold}
        self.threshold = threshold
        self.norm = norm[order]

    def l1_with_threshold(self, err):
        err = torch.abs(err)
        err = err-self.threshold
        err[err<0] = 0
        select_count = (err>0).sum()
        if select_count == 0:
            loss = err.sum()
        else:
            loss = err.sum()/select_count
        return loss
    
    def l2_with_threshold(self, err):
        err = err*err
        err_high = err>=self.threshold*self.threshold
        err_low = err<self.threshold*self.threshold
        loss_list = []
        if err_high.sum()>0:
            loss_high = err[err_high].mean()
            loss_list.append(loss_high)
        if err_low.sum()>0:
            loss_low = err[err_low].mean()
            loss_list.append(loss_low*0.01)
        loss = sum(loss_list)
        return loss

    def __call__(self, pred, label):
        err = pred-label
        return self.norm(err)
        


def get_pretrained_models(args_speaker):
    args_all = {"speaker":args_speaker}
    models = {}
    for key, args in args_all.items():
        CNN_arch = get_dict_from_args(['cnn_input_dim', 'cnn_N_filt', 'cnn_len_filt','cnn_max_pool_len',
                  'cnn_use_laynorm_inp','cnn_use_batchnorm_inp','cnn_use_laynorm','cnn_use_batchnorm',
                  'cnn_act','cnn_drop'], args.cnn)
    
        DNN_arch = get_dict_from_args(['fc_input_dim','fc_lay','fc_drop',
                'fc_use_batchnorm','fc_use_laynorm','fc_use_laynorm_inp','fc_use_batchnorm_inp',
                'fc_act'], args.dnn)
    
        Classifier = get_dict_from_args(['fc_input_dim','fc_lay','fc_drop', 
                  'fc_use_batchnorm','fc_use_laynorm','fc_use_laynorm_inp','fc_use_batchnorm_inp',
                  'fc_act'], args.classifier)
    
        CNN_arch['fs'] = args.windowing.fs
        model = SincClassifier(CNN_arch, DNN_arch, Classifier)
        if args.model_path!='none':
            print("load model from:", args.model_path)
            if os.path.splitext(args.model_path)[1] == '.pkl':
                checkpoint_load = torch.load(args.model_path)
                model.load_raw_state_dict(checkpoint_load)
            else:
                load_checkpoint(model, args.model_path, strict=True)

        model = model.cuda().eval()
        # freeze the model
        for p in model.parameters():
            p.requires_grad = False
        models[key] = model
        
    return models


def get_parser():
    parser = argparse.ArgumentParser("train generator")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234, help="")
    parser.add_argument("--eval", action='store_true', default=False, help="")
    parser.add_argument("--eval_every", type=int, default=-1, help="")
    parser.add_argument("--test", action='store_true', default=False, help="")
    parser.add_argument("--test_interpolation", action='store_true', default=False, help="")
    parser.add_argument("--beta", type=float, default=0, help="param for interpolation")
    parser.add_argument("--speaker_cfg", type=str, default="./config/timit_speaker_generator.cfg")
    parser.add_argument("--no_dist", action="store_true", default=True)
    parser.add_argument("--noise_dim", type=int, default=100, help="")
    parser.add_argument("--frame_dim", type=int, default=3200, help="")
    parser.add_argument("--noise_scale", type=float, default=1)
    parser.add_argument("--norm_clip", type=float, default=0.01)
    parser.add_argument("--speaker_factor", type=float, default=1)
    parser.add_argument("--norm_factor", type=float, default=1000)
    parser.add_argument("--wlen", type=int, default=200, help="length for a frame data, ms")
    # parser.add_argument("--fc_layers", type=int ,nargs='+', default=[1,1,0,0,0], help="fc layer float for generator model")
    parser.add_argument("--target", type=int, default=-1, help="")
    parser.add_argument("--pt_file", type=str, default='none', help="path for pretrained file")
    parser.add_argument("--data_root", type=str, default='./data/TIMIT/TIMIT_lower', help="path for data")
    parser.add_argument("--output_dir", type=str, default="./output/timit_generator")
    parser.add_argument("--dataset", choices=['timit', 'libri'], default='timit', help="the dataset name")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers for dataloader")
    parser.add_argument("--speaker_model", type=str, default="./output/SincNet_TIMIT/model_raw.pkl", help="path for pretrained speaker model")
    parser.add_argument("--test_output", type=str, default="./output/timit_generator_test", help="")
    parser.add_argument("--margin", type=int, default=100, help="margin for speaker loss")
    parser.add_argument("--mul", type=float, default=0.01, help="mul for speaker (target) loss")
    parser.add_argument("--epoch", type=int, default=-1, help="total training epoch")
    parser.add_argument("--norm_factor_grow", type=float, default=0, help="")
    args = parser.parse_args()
    return args


def _init_fn(work_id, seed):
    np.random.seed(seed+work_id)

def main(args):
    speaker_cfg = args.speaker_cfg
    args_speaker = read_conf(speaker_cfg, deepcopy(args))
    args_speaker.model_path = args.speaker_model
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("set seed: ", args_speaker.optimization.seed)
    torch.manual_seed(args_speaker.optimization.seed)
    np.random.seed(args_speaker.optimization.seed)
    random.seed(args_speaker.optimization.seed)

    torch.cuda.set_device(args.local_rank)
    if not args.no_dist:
        torch.distributed.init_process_group(backend="nccl")
    
    if args.dataset == 'timit':
        train_dataset = TIMIT_speaker_norm(args.data_root, train=True, wlen=args.wlen, phoneme=False, norm_factor=True, augment=False)
        test_dataset = TIMIT_speaker_norm(args.data_root, train=False, wlen=args.wlen, phoneme=False, norm_factor=True, augment=False)
    elif args.dataset == 'libri':
        train_dataset = LibriSpeech_speaker(args.data_root, train=True, wlen=args.wlen, phoneme=False, norm_factor=True, augment=False)
        test_dataset = LibriSpeech_speaker(args.data_root, train=False, wlen=args.wlen, phoneme=False, norm_factor=True, augment=False)
    else:
        raise ValueError
    
    pretrained_models = get_pretrained_models(args_speaker)

    loss_factors = {"speaker":args.speaker_factor, "norm":args.norm_factor}
    if args.target < 0: # non-targeted
        speaker_loss = SpeakerLoss(pretrained_models['speaker'], mul=args.mul, margin=args.margin)
    else: # targeted attack
        speaker_loss = SpeakerLossTarget(pretrained_models['speaker'], args.target, mul=args.mul, margin=args.margin)
    loss_all = {}
    loss_all['speaker'] = {'model':pretrained_models['speaker'], 'factor':loss_factors['speaker'], 'loss_func':speaker_loss, "factor_grow": 0}
    loss_all['norm'] = {'loss_func':MSEWithThreshold(args.norm_clip), 'factor':loss_factors['norm'], "factor_grow": args.norm_factor_grow}
    
    cost = UniversalLoss(loss_all)

    model = Generator1D(args.noise_dim, args.frame_dim)
    print(model)

    if args.pt_file!='none':
        print("load model from:", args.pt_file)
        if os.path.splitext(args.pt_file)[1] == '.pkl':
            checkpoint_load = torch.load(args.pt_file)
            model.load_raw_state_dict(checkpoint_load)
        else:
            load_checkpoint(model, args.pt_file)
        
    model = model.cuda()
    if args.eval:
        assert args.pt_file != 'none', "no pretrained model is provided!"
        print('only eval the model')
        evaluate(model, test_dataset, cost, args.target, args.noise_scale)
        return
    if args.test:
        assert args.pt_file != 'none', "no pretrained model is provided!"
        print("only test the model")
        if args.dataset == "timit":
            filename_list = open("./data/TIMIT/speaker/test.scp", 'r').readlines()
            filename_list = [_f.strip() for _f in filename_list]
            label_dict = np.load(os.path.join(args.data_root, "processed", "TIMIT_labels.npy"), allow_pickle=True).item()
            data_folder = args.data_root
        elif args.dataset == 'libri':
            filename_list = open(os.path.join(args.data_root, "libri_te.scp"), "r").readlines()
            filename_list = [_f.strip() for _f in filename_list]
            label_dict = np.load(os.path.join(args.data_root, "libri_dict.npy"), allow_pickle=True).item()
            data_folder = os.path.join(args.data_root, "Librispeech_spkid_sel")

        test_wav(model, filename_list, data_folder, args.test_output, pretrained_models['speaker'], label_dict, args.target, args.noise_scale)
        return

    if args.test_interpolation:
        assert args.pt_file != 'none', "no pretrained model is provided!"
        print("only test the model with interpolation")
        if args.dataset == "timit":
            filename_list = open("./data/TIMIT/speaker/test.scp", 'r').readlines()
            filename_list = [_f.strip() for _f in filename_list]
            label_dict = np.load(os.path.join(args.data_root, "processed", "TIMIT_labels.npy"), allow_pickle=True).item()
            data_folder = args.data_root
        elif args.dataset == 'libri':
            filename_list = open(os.path.join(args.data_root, "libri_te.scp"), "r").readlines()
            filename_list = [_f.strip() for _f in filename_list]
            label_dict = np.load(os.path.join(args.data_root, "libri_dict.npy"), allow_pickle=True).item()
            data_folder = os.path.join(args.data_root, "Librispeech_spkid_sel")

        test_interpolation(model, filename_list, data_folder, args.test_output, pretrained_models['speaker'], label_dict, args.target, args.noise_scale, args.beta)
        return
    

    print("train the model")
    batch_process = batch_process_generator
    eval_hook = EvalHook()
    optimizer = optim.Adam(model.parameters(), lr=args_speaker.optimization.lr, betas=(0.95, 0.999))
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 2, 1)
    if args.no_dist:
        train_kwarg = {'shuffle':True, 'worker_init_fn':partial(_init_fn, seed=args_speaker.optimization.seed)}
        test_kwarg = {'shuffle':True, 'worker_init_fn':partial(_init_fn, seed=args_speaker.optimization.seed)}
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_kwarg = {'shuffle':True, 'sampler':train_sampler, 'worker_init_fn':partial(_init_fn, seed=args_speaker.optimization.seed)}
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_kwarg = {'shuffle':True, 'sampler':test_sampler, 'worker_init_fn':partial(_init_fn, seed=args_speaker.optimization.seed)}
    train_dataloader = DataLoader(train_dataset, args_speaker.optimization.batch_size, num_workers=args.num_workers, pin_memory=True, **train_kwarg)
    test_dataloader = DataLoader(test_dataset, args_speaker.optimization.batch_size, num_workers=args.num_workers, pin_memory=True, **test_kwarg)
    eval_every = args.eval_every if args.eval_every else args_speaker.optimization.N_eval_epoch
    trainer = ClassifierTrainer(model, train_dataloader, optimizer, cost, batch_process, args.output_dir, 0, 
            test_dataloader, eval_hook=eval_hook, eval_every=eval_every, print_every=args_speaker.optimization.print_every, lr_scheduler=lr_scheduler,
            batch_param={'noise_scale':args.noise_scale, "target":args.target})
    trainer.logger.info(args)
    total_epoch = args.epoch if args.epoch>0 else args_speaker.optimization.N_epochs
    trainer.run(total_epoch)


if __name__=="__main__":
    args = get_parser()
    print(args)
    main(args)

