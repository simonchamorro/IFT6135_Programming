import os
import glob
import numpy as np
import matplotlib.pyplot as plt

logdir = './logs/'
plot_dir = './plots/'


def get_data(file):
    with open(file) as f:
        lines = f.readlines()
    
    if 'gpu' in file:
        return [float(l.split(' ')[0]) for l in lines]
    else:
        return [float(l) for l in lines]

def plot_results_lstm(file):
    exps = glob.glob(logdir + file, recursive=True)
    for exp in exps:
        train_loss_f = exp + '/train_loss.txt'
        valid_loss_f = exp + '/valid_loss.txt'
        train_ppl_f = exp + '/train_ppl.txt'
        valid_ppl_f = exp + '/valid_ppl.txt'
        train_time_f = exp + '/train_time.txt'
        test_ppl_f = exp + '/test_ppl.txt'

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

        # Print time and performance
        test_ppl = get_data(test_ppl_f)
        train_time = get_data(train_time_f)
        print('----------------')
        print('Exp name: ', exp)
        print('Train time: mean ', np.mean(train_time), ' std ', np.std(train_time))
        print('Valid Performance: ', valid_ppl)
        print('Test Performance: ', test_ppl)


def plot_results_vit(file):
    exps = glob.glob(logdir + file, recursive=True)
    for exp in exps:
        train_acc_f = exp + '/train_ppl.txt'
        valid_acc_f = exp + '/valid_ppl.txt'
        train_time_f = exp + '/train_time.txt'
        test_acc_f = exp + '/test_ppl.txt'
        gpu_f = exp + '/gpu_mem_usage.txt'

        # Accuracy curve over epochs
        train_acc = get_data(train_acc_f)
        valid_acc = get_data(valid_acc_f)
        plot_graph([train_acc, valid_acc], 
                    labels=['train', 'valid'], 
                    title='ViT Performance over Epochs',
                    save_as=exp.split('/')[-1] + '_acc_epochs.png',
                    axes=['Epoch', 'Accuracy'])

        # Accuracy curve over clock time
        train_acc = get_data(train_acc_f)
        valid_acc = get_data(valid_acc_f)
        train_time = get_data(train_time_f)
        plot_graph([train_acc, valid_acc], 
                    labels=['train', 'valid'], 
                    title='ViT Performance over Training Time',
                    save_as=exp.split('/')[-1] + '_acc_time.png',
                    axes=['Time (s)', 'Accuracy'],
                    x_vals=np.cumsum(train_time))

        # Print time and performance
        test_acc = get_data(test_acc_f)
        gpu = get_data(gpu_f)
        print('----------------')
        print('Exp name: ', exp)
        print('Train time: mean ', np.mean(train_time), ' std ', np.std(train_time))
        print('Train Performance: ', np.max(train_acc))
        print('Valid Performance: ', np.max(valid_acc))
        print('Test Performance: ', test_acc)
        print('GPU Memory: ', np.mean(gpu))


def plot_graph(data, labels=[], title='', save_as='', axes=[], x_vals=None):
    fig = plt.figure()
    for idx, d in enumerate(data):
        if x_vals is not None:
            plt.plot(x_vals, d,label=labels[idx], marker=".", markersize=10)
        else:
            plt.plot(d,label=labels[idx], marker=".", markersize=10)

    plt.ylabel(axes[1], fontsize=10)
    plt.xlabel(axes[0], fontsize=10)
    plt.title(title, fontsize=15)
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(plot_dir + save_as)


if __name__ == '__main__':    

    # LSTM exps
    plot_results_lstm('lstm_*')

    # ViT exps
    plot_results_vit('vit_*')