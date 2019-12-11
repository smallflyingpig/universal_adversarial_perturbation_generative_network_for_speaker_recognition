import torch
import torchvision
from torch.utils.data import Dataset
import os
import os.path as osp
import numpy as np 
import torchaudio
import scipy 
import pickle
import csv
import pandas as pd
import soundfile as sf


def read_list(filename):
    with open(filename, "r") as fp:
        data = fp.readlines()
        data = [_l.strip() for _l in data]
    return data

def load_pickle(filenmae, encoding='utf8'):
    with open(filenmae, mode="rb") as fp:
        data = pickle.load(fp, encoding=encoding)
    return data

class TIMIT(Dataset):
    def __init__(self, data_root, datalist_root, train=True, 
        fs=16000, cw_len=200, cw_shift=10, rand_amp_fact=0.2, oversampling=800):
        super(TIMIT, self).__init__()
        split = "train" if train else "test"
        self.data_root = data_root
        self.data_list = read_list(osp.join(datalist_root, "TIMIT_"+split+".scp"))
        self.label_dict = np.load(osp.join(datalist_root, "TIMIT_labels.npy"), allow_pickle=True).item()
        self.iterator = self.prepara_train_data if train else self.prepare_test_data
        self.wlen, self.wshift = int(fs*cw_len/1000.00), int(fs*cw_shift/1000.00)
        self.rand_amp_fact = rand_amp_fact
        self.oversampling = oversampling if train else 1

    @staticmethod
    def preprocess(wav_data, wrd_data):
        wav_data=torch.FloatTensor(wav_data)
        # signal noormallization
        wav_data=wav_data/wav_data.abs().max()
        # remove silences
        beg_sig=int(wrd_data[0].split(' ')[0])
        end_sig=int(wrd_data[-1].split(' ')[1])
        wav_data=wav_data[beg_sig:end_sig]
        return wav_data, wrd_data

    def __len__(self):
        return len(self.data_list)*self.oversampling

    def prepara_train_data(self, index):
        index = index % len(self.data_list)
        filename = self.data_list[index]
        [wav_data, fs] = torchaudio.load(osp.join(self.data_root, filename))
        wav_data = wav_data.squeeze(0)
        wrd_data = read_list(osp.join(self.data_root, osp.splitext(filename)[0]+".wrd"))
        wav_data, wrd_data = self.preprocess(wav_data, wrd_data)
        # random chunk
        rand_amp_arr = np.random.uniform(1.0-self.rand_amp_fact, 1.0+self.rand_amp_fact, 1)[0]
        wav_len = wav_data.shape[0]
        wav_beg = np.random.randint(int(wav_len-self.wlen-1))
        # data augmentation
        wav_data = wav_data[wav_beg:wav_beg+self.wlen]*rand_amp_arr
        label = self.label_dict[filename]
        return wav_data, label

    @staticmethod
    def chunk_wav(wav_data, label, wlen, wshift):
        wav_len = wav_data.shape[0]
        chunk_num = (wav_len-wlen)//wshift+1
        wav_data_chunk = torch.stack([wav_data[beg*wshift:beg*wshift+wlen] for beg in range(chunk_num)])
        # data augmentation
        label = torch.LongTensor([label]*chunk_num)
        return wav_data_chunk, label

    def prepare_test_data(self, index):
        filename = self.data_list[index]
        [wav_data, fs] = torchaudio.load(osp.join(self.data_root, filename))
        wav_data = wav_data.squeeze(0)
        wrd_data = read_list(osp.join(self.data_root, osp.splitext(filename)[0]+".wrd"))
        wav_data, wrd_data = self.preprocess(wav_data, wrd_data)
        label = self.label_dict[filename]
        wav_data_chunk, label = self.chunk_wav(wav_data, label, self.wlen, self.shift)
        return wav_data_chunk, label

    def __getitem__(self, index):
        return self.iterator(index)


class TIMIT_base(Dataset):
    def __init__(self):
        super(TIMIT_base, self).__init__()

    @staticmethod
    def preprocess(wav_data):
        norm_factor = np.abs(wav_data).max()
        wav_data = wav_data/norm_factor
        return wav_data, norm_factor

    def load_frame(self, wav_filename, offset, f_wlen):
        wav_data, fs = sf.read(wav_filename)
        # assert offset+f_wlen<=len(wav_data)
        wav_data, norm_factor =  self.preprocess(wav_data)# normlize
        offset = min(max(offset, 0), len(wav_data)-f_wlen)
        frame = wav_data[offset:offset+f_wlen]
        return frame, norm_factor
    
class TIMIT_speech(TIMIT_base):
    def __init__(self, data_root, train=True, fs=16000, wlen=200):
        super(TIMIT_speech, self).__init__()
        data_root_processed = osp.join(data_root, "processed")
        self.data_root = data_root
        self.fs, self.wlen = fs, wlen
        self.f_wlen = int(fs*wlen/1000)
        self.split = "train" if train else "test"
        self.augment = train
        # read csv and phone file
        table_file = osp.join(data_root_processed, "speaker_{}.csv".format(self.split))
        print("load table file from: ", table_file)
        self.data = pd.read_csv(table_file)
        print("table keys: ", self.data.keys())
        phone_file = osp.join(data_root_processed, "phone.pickle")
        print("load phone file from: ", phone_file)
        self.phoneme = load_pickle(phone_file)
        print("phone keys: ", self.phoneme.keys())
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index, :]
        frame, norm_factor = self.load_frame(osp.join(self.data_root, row['filename']), row['offset'], self.f_wlen)
        if self.augment:
           frame *= np.random.uniform(1-0.2, 1+0.2)
        label = row['phoneme']
        return frame, label


# with data augmentation now
class TIMIT_speaker(TIMIT_base):
    def __init__(self, data_root, train=True, fs=16000, wlen=200, wshift=10, phoneme=False, norm_factor=False, augment=True):
        super(TIMIT_speaker, self).__init__()
        data_root_processed = osp.join(data_root, "processed")
        self.data_root = data_root
        self.fs, self.wlen = fs, wlen
        self.f_wlen = int(fs*wlen/1000)
        self.f_wshift = int(fs*wshift/1000)
        self.split = "train" if train else "test"
        self.phoneme = phoneme
        self.norm_factor = norm_factor
        self.augment = augment
        # read csv and speaker id file
        table_file = osp.join(data_root_processed, "speaker_{}.csv".format(self.split))
        print("load table file from: ", table_file)
        self.data = pd.read_csv(table_file)
        print("table keys: ", self.data.keys())
        speaker_id_file = osp.join(data_root_processed, "speaker_id.pickle")
        print("load speaker id from: ", speaker_id_file)
        self.speaker_id = load_pickle(speaker_id_file)
        print("phone keys len: ", len(self.speaker_id))
        self.timit_labels = np.load(os.path.join(data_root_processed, "TIMIT_labels.npy"), allow_pickle=True).item()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index, :]
        if self.augment and self.split == 'train':
            offset = row['offset'] + np.random.randint(-self.f_wshift//2, self.f_wshift//2)
        else:
            offset = row['offset']

        frame, norm_factor = self.load_frame(
            osp.join(self.data_root, row['filename']), 
            offset, self.f_wlen)
        if self.augment and self.split == 'train':
            frame *= np.random.uniform(1-0.2, 1+0.2)
            # frame = np.clip(frame, -1, 1)
        speaker_id, phoneme = row['speaker_id'], row['phoneme']
        speaker_id = self.timit_labels[row['filename']]
        rtn = (frame, speaker_id)
        if self.phoneme:
            rtn += (phoneme,)
        if self.norm_factor:
            rtn += (norm_factor,)
        return rtn



# with data augmentation now
class TIMIT_speaker_norm(TIMIT_base):
    def __init__(self, data_root, train=True, fs=16000, wlen=200, wshift=10, phoneme=False, norm_factor=False, augment=True):
        super(TIMIT_speaker_norm, self).__init__()
        data_root_processed = osp.join(data_root, "processed")
        self.data_root = data_root
        self.fs, self.wlen = fs, wlen
        self.f_wlen = int(fs*wlen/1000)
        self.f_wshift = int(fs*wshift/1000)
        self.split = "train" if train else "test"
        self.phoneme = phoneme
        self.norm_factor = norm_factor
        self.augment = augment
        # read csv and speaker id file
        data_list_file = os.path.join(data_root_processed, "{}.scp".format(self.split))
        print("load data list file from: ", data_list_file)
        self.data = read_list(data_list_file)
        self.timit_labels = np.load(os.path.join(data_root_processed, "TIMIT_labels.npy"), allow_pickle=True).item()
        self.avg_frame_num = 100
    
    def __len__(self):
        return len(self.data)*self.avg_frame_num

    def __getitem__(self, index):
        index = index%len(self.data)
        filename = self.data[index]
        data_wav, fs = sf.read(osp.join(self.data_root,  filename))
        data_wav, norm_factor = self.preprocess(data_wav)
        data_len = len(data_wav)
        # print("data len: ", data_len)
        offset = np.random.randint(0, data_len-self.f_wlen-1)
        if self.augment and self.split == 'train':
            data_wav = np.random.uniform(1-0.2, 1+0.2) * data_wav[offset:offset+self.f_wlen]
            data_wav = np.clip(data_wav, -1, 1)
        else:
            data_wav = data_wav[offset:offset+self.f_wlen]
        speaker_id = self.timit_labels[filename]
        rtn = (data_wav, speaker_id)
        if self.phoneme:
            rtn += (None,)
        if self.norm_factor:
            rtn += (norm_factor, )
        return rtn

class LibriSpeech_speaker(Dataset):
    def __init__(self, data_root, train=True, fs=16000, wlen=200, wshift=10, phoneme=False, norm_factor=False, augment=True):
        super(LibriSpeech_speaker, self).__init__()
        data_root_processed = osp.join(data_root, 'processed')
        self.data_root = data_root
        self.fs, self.wlen = fs, wlen
        self.f_wlen = int(fs*wlen/1000)
        self.f_shift = int(fs*wshift/1000)
        self.split = "train" if train else "test"
        self.phoneme = phoneme
        self.norm_factor = norm_factor
        self.augment = augment
        scp_file = "libri_tr.scp" if train else "libri_te.scp"
        def read_list(filename):
            with open(filename, 'r') as fp:
                data = fp.readlines()
            data = [l.strip() for l in data]
            return data
        print("read data list form: ", scp_file)
        self.data = read_list(osp.join(data_root, scp_file))
        label_dict = osp.join(data_root, "libri_dict.npy")
        print("load label dict from: ", label_dict)
        self.label_dict = np.load(label_dict, allow_pickle=True).item()
        self.avg_frame_num = 100

    def __len__(self):
        return len(self.data)*self.avg_frame_num

    def __getitem__(self, index):
        index = index%len(self.data)
        filename = self.data[index]
        data_wav, fs = sf.read(osp.join(self.data_root,  "Librispeech_spkid_sel", filename))
        data_len = len(data_wav)
        # print("data len: ", data_len)
        offset = np.random.randint(0, data_len-self.f_wlen-1)
        if self.augment and self.split == 'train':
            data_wav = np.random.uniform(1-0.2, 1+0.2) * data_wav[offset:offset+self.f_wlen]
            # data_wav = np.clip(data_wav, -1, 1)
        else:
            data_wav = data_wav[offset:offset+self.f_wlen]
        speaker_id = self.label_dict[filename]
        rtn = (data_wav, speaker_id)
        if self.phoneme:
            rtn += (None,)
        if self.norm_factor:
            rtn += (1, )
        return rtn

