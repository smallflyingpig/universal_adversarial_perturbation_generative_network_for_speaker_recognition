import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import argparse
import librosa
import torch
import os 
import pandas as pd 

from model import BandPassFilter
from librosa import display

plt.rcParams["font.family"] = "Times New Roman"
axis_font_size = 13
annatation_font_size = 11

def plot_wave(filename, sr=16000, axis=False):
     data, sr = librosa.load(filename, sr)
     plt.figure()
     axis = "on" if axis else "off"
     plt.axis(axis)
     display.waveplot(data, sr)
     plt.show()

def plot_spectrogram(filename, sr=16000, axis=False):
     data, sr = librosa.load(filename, sr)
     D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
     plt.figure()
     axis = "on" if axis else "off"
     plt.axis(axis)
     display.specshow(D, sr=sr, y_axis='log')
     plt.colorbar(format="%+2.0f dB")
     plt.show()

def get_parser():
    parser = argparse.ArgumentParser("script")
    parser.add_argument("--func", type=str, choices=["plot_pert", "plot_bpf", "plot_freq_ana"], default="plot_freq_ana", help="")
    parser.add_argument("--plot_data_path", type=str, 
        default="./timit_freq_band_result.csv", 
        help="")

    args = parser.parse_args()
    return args

def plot_perturtation(data_path, label="frequency distribution"):
    plt.rcParams["font.size"] = axis_font_size
    data = np.load(data_path, allow_pickle=True)
    data_real = np.load(os.path.join(os.path.dirname(data_path), "real_datas.npy"), allow_pickle=True)
    data_phn1 = np.load(os.path.join(os.path.dirname(data_path), "pertutation_phn1.npy"), allow_pickle=True)
    width = 8000.0/1025
    def get_normed_spec(_data, _width):
        data_spec = [librosa.stft(d) for d in _data]
        data_spec_all = np.abs(np.concatenate(data_spec, axis=1)).sum(axis=1)
        data_spec_all_norm = data_spec_all/np.sum(data_spec_all*_width)
        return data_spec_all_norm
    data_spec_all_norm = get_normed_spec(data, width)
    data_real_spec_all_norm = get_normed_spec(data_real, width)
    data_phn1 = get_normed_spec(data_phn1, width)
    f, a = plt.subplots(figsize=(10, 5))
    center = [idx*width for idx in range(1025)]
    # a.bar(center, data_real_spec_all_norm, align='center', width=width*1.001, label="energy distribution, raw data", alpha=0.5, color='green')
    # a.bar(center, data_spec_all_norm, align='center', width=width*1.001, label="energy distribution, without phoneme restrict", alpha=0.5, color='red')
    # a.bar(center, data_phn1, align='center', width=width*1.001, label="energy distribution, with phoneme restrict", alpha=0.5, color='blue')
    a.bar(center, data_phn1, align='center', width=width*1.001, label="energy distribution, with phoneme restrict")

    a.set_xlabel('f/Hz')
    a.xaxis.set_label_coords(1.03, -0.025)
    # a.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # a.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # a.legend(prop={'size':22})
    a.legend()
    f.tight_layout()
    plt.show()

def plot_band_pass_filter(data_path, bpf_kernel_size=33):
    plt.rcParams["font.size"] = axis_font_size
    y, sr = librosa.load(data_path, sr=16000)
    plt.figure(figsize=(24, 6))
    plt.subplot(2,5,1)
    display.waveplot(y, sr=sr, x_axis='none')
    plt.title("raw waveform")
    # raw spectrogram
    plt.subplot(2,5,6)
    display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max), sr=sr, y_axis='linear')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Linear power spectrogram")
    for low_hz, plot_idx in zip([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000], [2,3,4,5, 7,8,9,10]):
        high_hz = low_hz+1000
        y_input = torch.tensor(y).unsqueeze(0).unsqueeze(0)
        bpf = BandPassFilter(kernel_size=bpf_kernel_size, stride=1, padding=bpf_kernel_size//2, low_hz=low_hz, high_hz=high_hz)
        y_pass = bpf.forward(y_input).squeeze().numpy()
        plt.subplot(2,5,plot_idx)

        display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y_pass)), ref=np.max), sr=sr)
        plt.colorbar(format="%+2.0f dB")
        plt.title("frequency band:[{}, {}]".format(low_hz, high_hz))
    plt.tight_layout()
    plt.show()

def plot_freq_analysis(data_path):
    plt.rcParams["font.size"] = axis_font_size
    data = pd.read_csv(data_path, index_col=False)
    data_timit_phn0 = data[data.apply(lambda x: x['dataset']=='timit' and x['phn']==0, axis=1)]
    data_timit_phn1 = data[data.apply(lambda x: x['dataset']=='timit' and x['phn']==1, axis=1)]
    data_libri_phn0 = data[data.apply(lambda x: x['dataset']=='libri' and x['phn']==0, axis=1)]

    # plot the curve
    data_all = [data_timit_phn1, data_timit_phn0, data_libri_phn0]
    rows = len(data_all)
    plt.figure(figsize=(18, 3*rows))
    for data_idx, (dataset_name, data_item) in enumerate(
        zip(["TIMIT, with phoneme restrict", "TIMIT, no phoneme restrict", "LibriSpeech, no phoneme restrict"], data_all[:2])):
        subplot_idx = data_idx*3+1
        plt.subplot(rows, 3, subplot_idx)
        x_list = [f*1000+500 for f in range(8)]
        y_list = data_item["SER_aligned"].tolist()
        anna_labels = ["[{}k, {}k]".format(f, f+1) for f in range(8)]
        anna_xy = list(zip(x_list, y_list))
        plt.plot(x_list, y_list, '-bd', markevery=[True for _ in x_list])

        # finetune the anna
        xytext_all = [(0,0) for _ in x_list]
        xytext_all[0] = (100, 0)
        xytext_all[1] = (140, 0)
        xytext_all[2] = (130, 0)
        xytext_all[3] = (-10, -10)
        xytext_all[4] = (-20, -30)
        xytext_all[5] = (-30, -5)
        xytext_all[6] = (0, -40)
        xytext_all[7] = (-10, -10)
        
        plt.rcParams["font.size"] = annatation_font_size
        for label, xy, xytext, x,y in zip(anna_labels, anna_xy, xytext_all, x_list, y_list):
            plt.annotate(label+",({:4d},{:5.2f})".format(x,y), xy=xy, ha='right', va='bottom', 
            xytext=(-10+xytext[0],10+xytext[1]), textcoords="offset points", bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
        plt.rcParams["font.size"] = axis_font_size
        plt.xlabel("f/Hz")
        plt.ylabel("SER/%")
        plt.title("({}) SER-f ({})".format(subplot_idx, dataset_name))

        xytext_all = [[(0,0) for _ in x_list], [(0,0) for _ in x_list]]
        xytext_all[0][0] = (100, 0)
        xytext_all[0][3] = (60, -40)
        xytext_all[0][4] = (60, -40)
        xytext_all[0][5] = (145, -10)
        xytext_all[0][6] = (140, -30)
        xytext_all[0][7] = (85, -30)

        xytext_all[1][7] = (-10, -10)
        xytext_all[1][6] = (30, -50)
        xytext_all[1][5] = (0, -40)
        xytext_all[1][4] = (-10, -20)
        xytext_all[1][3] = (-15, 0)
        xytext_all[1][2] = (130, -10)
        xytext_all[1][1] = (90, 0)
        xytext_all[1][0] = (120, 0)
        
        for plot_idx, (x_axis, x_label, xytext_item) in enumerate(zip(["SNR", "PESQ"], ["SNR/dB", "PESQ"], xytext_all)):
           subplot_idx = data_idx*3+plot_idx+2
           plt.subplot(len(data_all), 3, subplot_idx)
           anna_xy = list(zip(data_item[x_axis].tolist(), data_item["SER"].tolist()))
           plt.plot(x_axis, "SER", 'bd', data=data_item, markevery=[True for _ in data_item[x_axis].tolist()])
           # finetune the offset
           plt.rcParams["font.size"] = annatation_font_size
           for label, xy, xytext, x, y in zip(anna_labels, anna_xy, xytext_item, data_item[x_axis].tolist(), data_item['SER'].tolist()):
               plt.annotate(label+",({:5.2f},{:5.2f})".format(x,y), xy=xy, ha='right', va='bottom', 
                xytext=(-8+xytext[0],8+xytext[1]), textcoords="offset points", bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
           plt.rcParams["font.size"] = axis_font_size
           plt.xlabel(x_label)
           plt.ylabel("SER/%")
           plt.title("({}) SER-{} ({})".format(subplot_idx, x_axis, dataset_name))

    for data_idx, (dataset_name, data_item) in enumerate(
        zip(["LibriSpeech, no phoneme restrict"], data_all[2:])):
        data_idx += 2
        subplot_idx = data_idx*3+1
        plt.subplot(rows, 3, subplot_idx)
        x_list = [f*1000+500 for f in range(8)]
        y_list = data_item["SER_aligned"].tolist()
        anna_labels = ["[{}k, {}k]".format(f, f+1) for f in range(8)]
        anna_xy = list(zip(x_list, y_list))
        plt.plot(x_list, y_list, '-bd', markevery=[True for _ in x_list])

        # finetune the anna
        xytext_all = [(0,0) for _ in x_list]
        xytext_all[0] = (80, 0)
        xytext_all[1] = (140, -15)
        xytext_all[2] = (130, 0)
        xytext_all[3] = (-10, -10)
        xytext_all[4] = (-20, -10)
        xytext_all[5] = (-20, -5)
        xytext_all[6] = (30, -40)
        xytext_all[7] = (-10, -15)
        plt.rcParams["font.size"] = annatation_font_size
        for label, xy, xytext, x,y in zip(anna_labels, anna_xy, xytext_all, x_list, y_list):
            plt.annotate(label+",({:4d},{:5.2f})".format(x,y), xy=xy, ha='right', va='bottom', 
            xytext=(-8+xytext[0],8+xytext[1]), textcoords="offset points", bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
        plt.rcParams["font.size"] = axis_font_size
        plt.xlabel("f/Hz")
        plt.ylabel("SER/%")
        plt.title("({}) SER-f ({})".format(subplot_idx, dataset_name))

        xytext_all = [[(0,0) for _ in x_list], [(0,0) for _ in x_list]]
        xytext_all[0][0] = (100, 0)
        xytext_all[0][1] = (10, 20)
        xytext_all[0][3] = (60, -30)
        xytext_all[0][4] = (60, -40)
        xytext_all[0][5] = (125, -10)
        xytext_all[0][6] = (120, -30)
        xytext_all[0][7] = (85, -30)

        xytext_all[1][7] = (-10, -10)
        xytext_all[1][6] = (20, -30)
        xytext_all[1][5] = (-10, -10)
        xytext_all[1][4] = (-10, -20)
        xytext_all[1][3] = (-15, 0)
        xytext_all[1][2] = (120, 10)
        xytext_all[1][1] = (80, 30)
        xytext_all[1][0] = (110, 0)
        
        
        for plot_idx, (x_axis, x_label, xytext_item) in enumerate(zip(["SNR", "PESQ"], ["SNR/dB", "PESQ"], xytext_all)):
           subplot_idx = data_idx*3+plot_idx+2
           plt.subplot(len(data_all), 3, subplot_idx)
           anna_xy = list(zip(data_item[x_axis].tolist(), data_item["SER"].tolist()))
           plt.plot(x_axis, "SER", 'bd', data=data_item, markevery=[True for _ in data_item[x_axis].tolist()])
           # finetune the offset
           plt.rcParams["font.size"] = annatation_font_size
           for label, xy, xytext, x, y in zip(anna_labels, anna_xy, xytext_item, data_item[x_axis].tolist(), data_item['SER'].tolist()):
               plt.annotate(label+",({:5.2f},{:5.2f})".format(x,y), xy=xy, ha='right', va='bottom', 
                xytext=(-8+xytext[0],8+xytext[1]), textcoords="offset points", bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
           plt.rcParams["font.size"] = axis_font_size
           plt.xlabel(x_label)
           plt.ylabel("SER/%")
           plt.title("({}) SER-{} ({})".format(subplot_idx, x_axis, dataset_name))


    plt.tight_layout()
    plt.subplots_adjust(hspace=0.33, wspace=0.12)
    plt.show()
        

def main(args):
    if args.func == "plot_pert":
        plot_perturtation(args.plot_data_path)
    elif args.func == "plot_bpf":
        plot_band_pass_filter(args.plot_data_path)
    elif args.func == 'plot_freq_ana':
        plot_freq_analysis(args.plot_data_path)
    else:
        raise ValueError

if __name__=="__main__":
    args = get_parser()
    print(args)
    main(args)



