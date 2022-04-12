from cProfile import label
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_results(name='default'):
    with open('models_q3/' + name + '/pretraining_results.pkl', 'rb') as f:
        pretrain = pickle.load(f)
    with open('models_q3/' + name + '/classification_results.pkl', 'rb') as f:
        classification = pickle.load(f)    
    return pretrain, classification


if __name__ == '__main__':
    # Plot Q1
    with open('models_q1/results.pkl', 'rb') as f:
        data = pickle.load(f)
    
    elbo_scores = data['elbo_scores']
    elbo_scores = [-e for e in elbo_scores]
    log_p_x = data['log_p_x']

    plt.figure()
    plt.plot(elbo_scores, marker=".", markersize=10)
    plt.ylabel('ELBO', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.title('ELBO Score over Epochs', fontsize=15)
    plt.grid()
    plt.savefig('plots/q1_elbo.png')

    # Plot Q3
    pretrain, classification = load_results('default')
    no_stop_grad_pre, no_stop_grad_class = load_results('no-stop-grad')
    no_pred_mlp_pre, no_pred_mlp_class = load_results('no-pred-mlp')
    fixed_random_init_pre, fixed_random_init_class = load_results('fixed-random-init')

    plt.figure()
    plt.plot(pretrain['pretraining_losses'], marker=".", markersize=5, label='default')
    plt.plot(no_stop_grad_pre['pretraining_losses'], marker=".", markersize=5, label='no-stop-grad')
    plt.plot(no_pred_mlp_pre['pretraining_losses'], marker=".", markersize=5, label='no-pred-mlp')
    plt.plot(fixed_random_init_pre['pretraining_losses'], marker=".", markersize=5, label='fixed-random-init')
    plt.ylabel('Pretraining Loss', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.title('Pretraining Loss over Epochs', fontsize=15)
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig('plots/q3_pretraining_loss.png')

    plt.figure()
    plt.plot(pretrain['pretraining_knn_accs'], marker=".", markersize=5, label='default')
    plt.plot(no_stop_grad_pre['pretraining_knn_accs'], marker=".", markersize=5, label='no-stop-grad')
    plt.plot(no_pred_mlp_pre['pretraining_knn_accs'], marker=".", markersize=5, label='no-pred-mlp')
    plt.plot(fixed_random_init_pre['pretraining_knn_accs'], marker=".", markersize=5, label='fixed-random-init')
    plt.ylabel('KNN Accuracy', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.title('KNN Accuracy over Epochs', fontsize=15)
    plt.grid()
    plt.legend(loc='lower right')
    plt.savefig('plots/q3_knn_acc.png')

    plt.figure()
    plt.plot(classification['acc1_val'], marker=".", markersize=5, label='default')
    plt.plot(no_stop_grad_class['acc1_val'], marker=".", markersize=5, label='no-stop-grad')
    plt.plot(no_pred_mlp_class['acc1_val'], marker=".", markersize=5, label='no-pred-mlp')
    plt.plot(fixed_random_init_class['acc1_val'], marker=".", markersize=5, label='fixed-random-init')
    plt.ylabel('Accuracy', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.title('Classification Accuracy over Epochs', fontsize=15)
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('plots/q3_acc.png')

    