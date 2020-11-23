"""
prepare data for speech/speaker recognition
"""
import os 
import argparse
import soundfile as sf 
import numpy as np 
from pathlib import Path
import tqdm 
import pickle
import csv
import pandas as pd 
"""
phonemeset, fork from: https://github.com/matthijsvk/TIMITspeech
"""
# using the 39 phone set proposed in (Lee & Hon, 1989)
# Table 3. Mapping from 61 classes to 39 classes, as proposed by Lee and Hon, (Lee & Hon,
# 1989). The phones in the left column are folded into the labels of the right column. The
# remaining phones are left intact.
import logging

logger_phonemeSet = logging.getLogger('phonemeSet')
logger_phonemeSet.setLevel(logging.ERROR)

phoneme_set_61_39 = {
    'ao': 'aa',  # 1
    'ax': 'ah',  # 2
    'ax-h': 'ah',
    'axr': 'er',  # 3
    'hv': 'hh',  # 4
    'ix': 'ih',  # 5
    'el': 'l',  # 6
    'em': 'm',  # 6
    'en': 'n',  # 7
    'nx': 'n',
    'eng': 'ng',  # 8
    'zh': 'sh',  # 9
    "ux": "uw",  # 10
    "pcl": "sil",  # 11
    "tcl": "sil",
    "kcl": "sil",
    "qcl": "sil",
    "bcl": "sil",
    "dcl": "sil",
    "gcl": "sil",
    "h#": "sil",
    "#h": "sil",
    "pau": "sil",
    "epi": "sil",
    "q": "sil",
}

# from https://www.researchgate.net/publication/275055833_TCD-TIMIT_An_audio-visual_corpus_of_continuous_speech
phoneme_set_39_list = [
    'iy', 'ih', 'eh', 'ae', 'ah', 'uw', 'uh', 'aa', 'ey', 'ay', 'oy', 'aw', 'ow',  # 13 phns
    'l', 'r', 'y', 'w', 'er', 'm', 'n', 'ng', 'ch', 'jh', 'dh', 'b', 'd', 'dx',  # 14 phns
    'g', 'p', 't', 'k', 'z', 'v', 'f', 'th', 's', 'sh', 'hh', 'sil'  # 12 pns
]
values = [i for i in range(0, len(phoneme_set_39_list))]
phoneme_set_39 = dict(zip(phoneme_set_39_list, values))
classToPhoneme39 = dict((v, k) for k, v in phoneme_set_39.items())

# from http://www.intechopen.com/books/speech-technologies/phoneme-recognition-on-the-timit-database, page 5
phoneme_set_61_list = [
    'iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr',
    'ax-h', 'jh',
    'ch', 'b', 'd', 'g', 'p', 't', 'k', 'dx', 's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng', 'em', 'nx',
    'en', 'eng', 'l', 'r', 'w', 'y', 'hh', 'hv', 'el', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'q', 'pau', 'epi',
    'h#',
]
values = [i for i in range(0, len(phoneme_set_61_list))]
phoneme_set_61 = dict(zip(phoneme_set_61_list, values))


def convertPredictions(predictions, phoneme_list=classToPhoneme39, valid_frames=None, outputType="phonemes"):
    # b is straight conversion to phoneme chars
    predictedPhonemes = [phoneme_list[predictedClass] for predictedClass in predictions]

    # c is reduced set of b: duplicates following each other are removed until only 1 is left
    reducedPhonemes = []
    for j in range(len(predictedPhonemes) - 1):
        if predictedPhonemes[j] != predictedPhonemes[j + 1]:
            reducedPhonemes.append(predictedPhonemes[j])

    # get only the outputs for valid phrames
    validPredictions = [predictedPhonemes[frame] for frame in valid_frames]

    # return class outputs
    if outputType != "phonemes":
        predictedPhonemes = [phoneme_set_39[phoneme] for phoneme in predictedPhonemes]
        reducedPhonemes = [phoneme_set_39[phoneme] for phoneme in reducedPhonemes]
        validPredictions = [phoneme_set_39[phoneme] for phoneme in validPredictions]

    return predictedPhonemes, reducedPhonemes, validPredictions

# generate new .phn file with mapped phonemes (from 61, to 39 -> see dictionary in phoneme_set.py)
def transformPhn(phn_file):
    # extract label from phn
    phn_labels = []
    with open(phn_file, 'r') as csvfile:
        phn_reader = csv.reader(csvfile, delimiter=' ')
        for row in phn_reader:
            start, stop, label = row[0], row[1], row[2]
            if label not in phoneme_set_39.keys():  # map from 61 to 39 phonems using dict
                label = phoneme_set_61_39.get(label)
            classNumber = phoneme_set_39[label] # get class number
            phn_labels.append([int(start), int(stop), classNumber])
    return phn_labels

def chunk_file_with_phone(wav_data, phn_label, fs, wlen, wshift):
    wlen = int((fs*wlen)/1000) # ms to hz
    wshift = int((fs*wshift)/1000) # ms to hz
    
    offsets = []
    labels = []

    cur_idx = 0
    beg_sample = phn_label[0][0]
    length = min(len(wav_data), phn_label[-1][1])
    while beg_sample+wlen<length and cur_idx<len(phn_label):
        center = beg_sample+wlen//2
        if center<phn_label[cur_idx][1]:
            label = phn_label[cur_idx][2]
            offset = beg_sample
            labels.append(label)
            offsets.append(offset)
            beg_sample += wshift
        else:
            cur_idx += 1
    assert len(offsets)==len(labels)
    return offsets, labels

def prepare_data_for_speech(data_root, output_dir, fs=16000, wlen=200, wshift=10):
    """
    fs: Hz, 
    wlen: ms, 
    wshift: ms
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    split = ['train', 'test']
    file_num = {'train':4620, 'test':1680}
    for s  in split:
        print("process {}ing set".format(s))
        todo = list(Path(os.path.join(data_root,s)).rglob(r"*.[wW][aA][vV]"))
        todo = [_t for _t in todo if len(str(_t).split('.'))<3] # filter out the file with ext .wav.wav
        print(len(todo),'audio files found in {}ing set (should be {})'.format(s, file_num[s]))
        bar = tqdm.tqdm(todo)
        offsets = []
        labels = []
        filenames = []
        for f in bar:
            phn_f = os.path.splitext(f)[0]+'.phn'
            phn_label = transformPhn(phn_f)
            wav_data, fs_tmp = sf.read(f)
            assert fs_tmp == fs
            wav_data.astype(float)
            offsets_this, labels_this = chunk_file_with_phone(wav_data, phn_label, fs, wlen, wshift)
            del wav_data
            offsets.append(offsets_this)
            labels.append(labels_this)
            filenames.append(['/'.join(str(f).split('/')[-3:])]*len(labels_this))
        filenames, offsets, labels = np.concatenate(filenames), np.concatenate(offsets), np.concatenate(labels)
        bar.close()

        table = pd.DataFrame({"filename":filenames, "offset":offsets, 'label':labels})
        table.to_csv(os.path.join(output_dir, "speech_{}.csv".format(s)))
    # dump the phone
    with open(os.path.join(output_dir, "phone.pickle"), "wb") as fp:
        pickle.dump({"phoneme_set_61_39":phoneme_set_61_39, "phoneme_set_39":phoneme_set_39}, fp)
    print("prepare dataset for speech recognition end, save data into {}".format(output_dir))



def prepare_data_for_speaker(data_root, output_dir, fs=16000, wlen=200, wshift=10):
    split = ['train', 'test']
    data_list_dir = "./data/TIMIT/speaker"
    # get speaker id set
    with open(os.path.join(data_list_dir, "train.scp"), 'r') as fp:
        train_filenames = fp.readlines()
    speaker_ids_train = sorted(list(set( [_f.split('/')[-2] for _f in train_filenames])))
    speaker_id_set = {_id:v for _id, v in zip(speaker_ids_train, range(len(speaker_ids_train)))}
    print("total speaker num:", len(speaker_id_set))
    for s in split:
        with open(os.path.join(data_list_dir, "{}.scp".format(s)), "r") as fp:
            todo = fp.readlines()
        todo = [_t.strip() for _t in todo]
        print(len(todo), " audio file found in {}ing list".format(s))
        bar = tqdm.tqdm(todo)
        filenames, offsets, speaker_ids, phonemes = [], [], [], []
        for f in bar:
            phn_f = os.path.join(data_root, os.path.splitext(f)[0]+'.phn') 
            phn_label = transformPhn(phn_f)
            wav_data, fs_tmp = sf.read(os.path.join(data_root, f))
            assert fs_tmp == fs
            wav_data.astype(float)
            offsets_this, phonemes_this = chunk_file_with_phone(wav_data, phn_label, fs, wlen, wshift)
            del wav_data
            filenames.append([f]*len(phonemes_this))
            offsets.append(offsets_this)
            phonemes.append(phonemes_this)
            speaker_ids.append([speaker_id_set[f.split('/')[-2]]]*len(phonemes_this))
        filenames, offsets = np.concatenate(filenames), np.concatenate(offsets)
        speaker_ids, phonemes = np.concatenate(speaker_ids), np.concatenate(phonemes)
        bar.close()
        table = pd.DataFrame({'filename':filenames, 'offset':offsets, 'speaker_id':speaker_ids, 'phoneme':phonemes})
        table.to_csv(os.path.join(output_dir, "speaker_{}.csv".format(s)))
    # dump the speaker id set
    with open(os.path.join(output_dir, "speaker_id.pickle"), "wb") as fp:
        pickle.dump(speaker_id_set, fp)
    print("prepare dataset for speaker recognition end, save data into {}".format(output_dir))
    


def get_parser():
    parser = argparse.ArgumentParser("prepare dataset")
    parser.add_argument("--type", choices=['speech', 'speaker'], default='speaker', help="prepare data for speech or speaker")
    parser.add_argument("--data_root", type=str, default="./data/TIMIT/TIMIT_lower", help="root for TIMIT dataset")
    parser.add_argument("--output_dir", type=str, default="./data/TIMIT/TIMIT_lower/processed", help="")
    parser.add_argument("--fs", type=int, default=16000, help="sample rate")
    parser.add_argument("--wlen", type=int, default=200, help="ms")
    parser.add_argument("--wshift", type=int, default=10, help="ms")
    args = parser.parse_args()
    return args

def main(args):
    if args.type == 'speech':
        prepare_data_for_speech(args.data_root, args.output_dir, args.fs, args.wlen, args.wshift)
    elif args.type == 'speaker':
        prepare_data_for_speaker(args.data_root, args.output_dir, args.fs, args.wlen, args.wshift)
    else:
        raise NotImplementedError

if __name__=="__main__":
    args = get_parser()
    main(args)



