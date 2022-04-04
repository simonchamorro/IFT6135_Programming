import pickle
import numpy as np
import matplotlib.pyplot as plt



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
    with open('q3_solution.pkl', 'rb') as f:
        data = pickle.load(f)
    