"""Plot the decodability of the pure inputs in various baseline models.

The used baselines are 
    1. linear decoding, 
    2. k-means clustering, 
    3. Slow Feature Analysis,
    4. A predictive coding network trained on the static input images

Example:
    $ python train.py --data smallnorb_extended.npy --labels smallnorb_labels.npy --trafos 0 0 0 --resultfolder fig5a --epochs 20 --n_runs 4

    $ python train.py --data mnist_extended_fast.npy --labels labels_mnist_extended.npy --trafos 0 0 0 --resultfolder fig5b_trafo-0_static-0_noise-0 --epochs 20 --n_runs 4
    $ python train.py --data mnist_extended_fast.npy --labels labels_mnist_extended.npy --trafos 1 1 1 --resultfolder fig5b_trafo-1_static-0_noise-0 --epochs 20 --n_runs 4
    $ python train.py --data mnist_extended_fast.npy --labels labels_mnist_extended.npy --trafos 2 2 2 --resultfolder fig5b_trafo-2_static-0_noise-0 --epochs 20 --n_runs 4

    $ python train.py --data mnist_extended_fast.npy --labels labels_mnist_extended.npy --trafos 0 0 0 --resultfolder fig5b_trafo-0_static-1_noise-0 --do_train_static 1 --epochs 20 --n_runs 4
    $ python train.py --data mnist_extended_fast.npy --labels labels_mnist_extended.npy --trafos 1 1 1 --resultfolder fig5b_trafo-1_static-1_noise-0 --do_train_static 1 --epochs 20 --n_runs 4
    $ python train.py --data mnist_extended_fast.npy --labels labels_mnist_extended.npy --trafos 2 2 2 --resultfolder fig5b_trafo-2_static-1_noise-0 --do_train_static 1 --epochs 20 --n_runs 4

    $ python post_run_analysis/plot_fig5ab.py 
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from scripts.decoding_functions.decoding import kmeans_decoding, sfa_decoding, linear_decodability
from tabulate import tabulate

def plot_fig5a():
    """Plot the linear readout accuracy of the continuously trained network and various baselines.
     
    """
    # Load data for baseline computation
    train_array = np.load('../data/data_preprocessed/smallnorb_extended.npy')
    train_labels = np.load('../data/data_preprocessed/smallnorb_labels.npy')

    # Compute baselines
    kmeans_accuracy_smallnorb_train, kmeans_accuracy_smallnorb_validation = kmeans_decoding(
        train_array=train_array,
        train_labels=train_labels,
        n_clusters=5
    )
    sfa_accuracy_smallnorb_train, sfa_accuracy_smallnorb_validation = sfa_decoding(
        train_array=train_array,
        train_labels=train_labels,
        n_components=10
    )

    n_classes, n_frames_per_sequence = train_array.shape[0], train_array.shape[3] # Determine some dimensions of the dataset to reshape it appropriately
    linear_decoding_accuracy_train = linear_decodability(
        train_array.reshape((n_classes, n_frames_per_sequence, -1)), # Linear decoding requires specific format
        train_labels,
        n_splits=3
    )

    # Prepare loading of metrics
    metrics_path = '../results/intermediate/fig5a/metrics'
    acc_means = [] # decoding accuracy
    acc_std = [] # standard deviation of accuracy across runs
    epochs = [] 

    # Load means
    with open(metrics_path + '/linear_decoding_accuracy_train_classes_smallnorb_extended_noise-0_static-0.txt', 'rb') as fp:
        accuracy_data = pickle.load(fp) 
        acc_means.append(accuracy_data)
        epochs.append(range(len(accuracy_data)))
    
    # Load standard deviations
    with open(metrics_path + '/linear_decoding_accuracy_train_classes_std_smallnorb_extended_noise-0_static-0.txt', 'rb') as fp:
        accuracy_std_data = pickle.load(fp) 
        acc_std.append(accuracy_std_data)
        epochs.append(range(len(accuracy_std_data)))

    # Plot
    colors = ['tab:purple','black','black', 'black', 'black']
    fig, ax = plt.subplots()
    plt.errorbar(epochs[0], acc_means[0], acc_std[0], label='PC-continuous', color=colors[0], linewidth=2.5)
    plt.axhline(y=0.2, linewidth=2, label='Chance level', color=colors[1], linestyle='-')
    plt.axhline(y=kmeans_accuracy_smallnorb_train, linewidth=2, label='k-means inputs', color=colors[2], linestyle='--')
    plt.axhline(y=sfa_accuracy_smallnorb_train, linewidth=2, label='SFA inputs', color=colors[3], linestyle='-.')
    plt.axhline(y=linear_decoding_accuracy_train, linewidth=2, label='LD inputs', color=colors[4], linestyle=':')
    plt.xlabel('Training epoch', fontsize=label_fontsize)
    plt.ylabel('Decoding accuracy', fontsize=label_fontsize)
    plt.ylim([0, 1.05])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
    plt.legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=legend_fontsize)
    plt.tight_layout()

    # Save figure
    figures_path = '../results/figures/fig5/'
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)
    plt.savefig(figures_path + 'fig5a.svg')
    
def plot_fig5b():
    """Plot the linear readout accuracy of the continuously trained network and the statically trained network.
    
    """
    # Load intermediate results from training run
    path_intermediate_results = '../results/intermediate'
    acc_means = [] # decoding accuracy
    acc_std = [] # standard deviation across randomly seeded runs
    for do_train_static in [0,1]:
        for trafoIt in [0,1,2]:
            with open(f'{path_intermediate_results}/fig5b_trafo-{trafoIt}_static-{do_train_static}_noise-0/metrics/linear_decoding_accuracy_train_classes_mnist_extended_fast_noise-0_static-{do_train_static}.txt', "rb") as fp:
                    accuracy_data=pickle.load(fp) 
                    acc_means.append(accuracy_data)
            with open(f'{path_intermediate_results}/fig5b_trafo-{trafoIt}_static-{do_train_static}_noise-0/metrics/linear_decoding_accuracy_train_classes_std_mnist_extended_fast_noise-0_static-{do_train_static}.txt', "rb") as fp:
                    accuracy_std_data = pickle.load(fp) 
                    acc_std.append(accuracy_std_data)

    # Plot
    fig, ax = plt.subplots()
    plt.title(' ', fontsize = title_fontsize)
    dataset_labels = ['Fast translation', 'Fast rotation', 'Scaling', 'Fast translation static', 'Fast rotation static', 'Scaling static']
    colors = ['tab:green','tab:orange','tab:blue', 'k']
    for i in range(3):
        plt.errorbar(range(len(acc_means[i])), acc_means[i], acc_std[i], label=dataset_labels[i], color=colors[i])
    for i in range(3,6):
        plt.errorbar(range(len(acc_means[i])), acc_means[i], acc_std[i], label=dataset_labels[i-3] +' static', color=colors[i-3], ls='--')
    
    ax.axhline(y=0.1, linewidth=1, label='Chance level', color=colors[3])
    ax.set_xlabel('Training epoch', fontsize = label_fontsize)
    ax.set_ylabel('Decoding accuracy', fontsize = label_fontsize)
    ax.set_ylim([0, 1])
    ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
    ax.legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=legend_fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()

    # Save figure
    figures_path = '../results/figures/fig5/'
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)
    plt.savefig(figures_path + 'fig5b.svg')

if __name__ == '__main__':
    # Shared parameters for both panels.
    label_fontsize = 16
    title_fontsize = label_fontsize + 3
    ticklabel_fontsize = label_fontsize 
    legend_fontsize = label_fontsize - 2

    # Plot the two panels.
    plot_fig5a()
    plot_fig5b()
    plt.show()
