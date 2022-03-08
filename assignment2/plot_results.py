import os
import glob
import numpy as np
import matplotlib.pyplot as plt

logdir = './logs/'
plot_dir = './plots/'


def get_data(file):
    with open(file) as f:
        lines = f.readlines()
    
    return [float(l) for l in lines]

def plot_results_lstm(file):
    exps = glob.glob(logdir + file, recursive=True)
    for exp in exps:
        train_loss_f = exp + '/train_loss.txt'
        valid_loss_f = exp + '/valid_loss.txt'
        train_ppl_f = exp + '/train_ppl.txt'
        valid_ppl_f = exp + '/valid_ppl.txt'

        # Loss curve
        train_loss = get_data(train_loss_f)
        valid_loss = get_data(valid_loss_f)
        plot_graph([train_loss, valid_loss], 
                    labels=['train', 'valid'], 
                    title='LSTM Loss Curve',
                    save_as=exp.split('/')[-1] + '_loss.png',
                    axes=['Epoch', 'Loss'])

        # PPL curve
        train_ppl = get_data(train_ppl_f)
        valid_ppl = get_data(valid_ppl_f)
        plot_graph([train_ppl, valid_ppl], 
                    labels=['train', 'valid'], 
                    title='LSTM Performance Curve',
                    save_as=exp.split('/')[-1] + '_ppl.png',
                    axes=['Epoch', 'Perplexity'])


def plot_graph(data, labels=[], title='', save_as='', axes=[]):
    fig = plt.figure()
    for idx, d in enumerate(data):
        plt.plot(d,label=labels[idx], marker=".", markersize=10)

    plt.ylabel(axes[1], fontsize=10)
    plt.xlabel(axes[0], fontsize=10)
    plt.title(title, fontsize=15)
    plt.legend(loc='lower left')
    plt.grid()
    plt.savefig(plot_dir + save_as)


if __name__ == '__main__':    

    # LSTM exps
    test = plot_results_lstm('lstm_*')