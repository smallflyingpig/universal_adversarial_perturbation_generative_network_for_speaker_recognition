#### plot the figure for paper
import matplotlib.pyplot as plt 
import pandas as pd 
import argparse

plt.rcParams["font.family"] = "Times New Roman"
axis_font_size = 13
annatation_font_size = 11

def plot_UAP_length(args):
    plt.rcParams["font.size"] = axis_font_size
    data = pd.read_csv(args.data_path)
    timit_data = data[data.apply(lambda x: x['dataset']=='timit', axis=1)]
    noise_data = data[data.apply(lambda x: x['dataset']=='noise', axis=1)]
    # libri_data = data[data.apply(lambda x: x['dataset']=='libri', axis=1)]
    # data_all = [timit_data, libri_data]
    data_all = [timit_data]
    x_key_list = ['SNR', 'PESQ']
    plt.figure(figsize=(10, 4))
    for row_idx, (dataset_name, _d) in enumerate(zip(['TIMIT', 'LibriSpeech'], data_all)):
        for col_idx, x_key in enumerate(x_key_list):
            subplot_idx = row_idx*len(data_all)+col_idx+1
            plt.subplot(len(data_all),len(x_key_list),subplot_idx)
            wlen_list = [200, 400, 600, 800]
            color_list = ['r', 'y', 'g', 'b']
            for wlen, color in zip(wlen_list, color_list):
                data_wlen = _d[_d.apply(lambda x: x['wlen']==wlen, axis=1)]
                data_wlen = data_wlen.sort_values(x_key)
                if len(data_wlen)<1:
                    continue
                x, y = data_wlen[x_key].tolist(), data_wlen['SER'].tolist()
                plt.plot(x_key,'SER','-'+color+'d', data=data_wlen, label='UAP len='+str(wlen))
            # plot noise baseline
            noise_data = noise_data.sort_values(x_key)
            plt.plot(x_key, 'SER', '-'+'c'+'d', data=noise_data, label='random noise')
            plt.xlabel(x_key)
            plt.ylabel("SER")
            plt.legend(loc=3)
            plt.title("({}) SER-{} ({})".format(subplot_idx, x_key, dataset_name))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.show()



def get_parser():
    parser = argparse.ArgumentParser("plot for UAPs")
    parser.add_argument("--data_path", type=str, default="./UAP_ablation_study_data.csv", help="")
    parser.add_argument("--plot_type", choices=['UAP_length'], default='UAP_length', help="")
    args = parser.parse_args()
    return args

def main(args):
    function_all = {'UAP_length':plot_UAP_length}
    function_all.get(args.plot_type, lambda x: print("plot type error"))(args)


if __name__ == "__main__":
    args = get_parser()
    main(args)