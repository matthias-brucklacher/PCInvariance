"""Figure 5. Show upscaling behavior for multiple network types.

Baseline performance is computed here, network performance is loaded from training runs as described in the README.

"""
from scripts.decoding_functions.decoding import sfa_decoding, test_generalization, kmeans_decoding, linear_decodability
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def load_network_accuracy(resultfolder, architectures, n_ipc_list, decode_mode):
    """Opens files in given directory and returns mean and std across runs for different dataset sizes.

    Args:
        resultfolder (str): File location.
        Architectures (dict): Dictionary with integer values specifiying the number of neurons in each area.
        n_ipc_list (list): List of instance-per-class values (the higher, the larger the dataset was).
        decode_mode (str): 'validate' or 'train_classes'.
    
    Returns:
        acc_means (dict): Linear decoding accuracy averaged across differently initialized runs.
        acc_std (dict): Standard deviation of linear decoding accuracy across runs.

    """
    acc_means = {}
    acc_std = {}
    for networkIt in ['net_1', 'net_2']:
        acc_means.update({networkIt: []}) 
        acc_std.update({networkIt: []}) 
        for n_ipcIt in n_ipc_list:
            with open(resultfolder+f'fig5cd_arch-{architectures[networkIt]}_nipc-{n_ipcIt}/metrics/linear_decoding_accuracy_{decode_mode}_mnist_extended_fast_noise-0_static-None.txt', "rb") as fp:
                list = pickle.load(fp)
                acc_means[networkIt].append(list[-1]) # take value at last training epoch
            with open(resultfolder+f'fig5cd_arch-{architectures[networkIt]}_nipc-{n_ipcIt}/metrics/linear_decoding_accuracy_{decode_mode}_std_mnist_extended_fast_noise-0_static-None.txt', "rb") as fp:
                list = pickle.load(fp)
                acc_std[networkIt].append(list[-1]) # take value at last training epoch
    return acc_means, acc_std

def make_input_data_3d(data):
    """Flattens the first three and the last two dimensions of a five-dimensional array.

    Args:
        data (numpy.ndarray or torch.Tensor): Input data. Shape is (n_classes, n_instances_per_class, n_trafos, n_frames_per_sequence, width, height) 
    
    """
    shape = data.shape
    return data.reshape(shape[0] * shape[1] * shape[2], shape[3], shape[4] * shape[5])

if __name__ == '__main__':
    # Load data for baseline models
    trafos = [0] # Used transformations
    data = torch.tensor(np.load('../data/data_preprocessed/mnist_extended_fast.npy'))[:, :, trafos] # Shape is (10, 50, 1, 6, 34, 34)
    data_labels = torch.tensor(np.load('../data/data_preprocessed/labels_mnist_extended.npy'))[:, :, trafos] # Shape is (10, 50, 6)
    n_instances_per_class_validate = 20
    test_data = data[:, -n_instances_per_class_validate:]
    test_labels = data_labels[:, -n_instances_per_class_validate:]

    # Compute baseline levels
    sfa_decoding_accuracy_train, sfa_decoding_accuracy_validate = [], []
    kmeans_decoding_accuracy_train, kmeans_decoding_accuracy_validate = [], []
    linear_decoding_accuracy_train, linear_decoding_accuracy_validate = [], []

    n_ipc_list = [1, 5, 10, 15, 20] # instances-per-class settings
    n_samples_list = [i*10 for i in n_ipc_list]
    for n_ipc in n_ipc_list:
        # Baseline 1: SFA
        sfa_train, sfa_validate = sfa_decoding(
            train_array=data[:, :n_ipc],
            train_labels=data_labels[:, :n_ipc],
            validate_array=test_data,
            validate_labels=test_labels,
            n_components=30
        )
        sfa_decoding_accuracy_train.append(sfa_train)
        sfa_decoding_accuracy_validate.append(sfa_validate)

        # Baseline 2: K-means
        # kmeans_train, kmeans_validate = kmeans_decoding(
        #     train_array=data[:, [n_ipc]],
        #     train_labels=data_labels[:, [n_ipc]],
        #     validate_array=test_data,
        #     validate_labels=test_labels,
        #     n_clusters=10
        # )
        # kmeans_decoding_accuracy_train.append(kmeans_train)
        # kmeans_decoding_accuracy_validate.append(kmeans_validate)

        # Baseline 3: Linear readout
        linear_decoding_accuracy_train.append( 
            linear_decodability(
                make_input_data_3d(data[:, :n_ipc]), 
                data_labels[:, :n_ipc],
                n_splits = 3) 
        )

        linear_decoding_accuracy_validate.append(
            test_generalization( # reps_train, labels_train, reps_validate, labels_validate
                reps_train=make_input_data_3d(data[:, :n_ipc]),
                labels_train=data_labels[:, :n_ipc],
                reps_validate=make_input_data_3d(test_data),
                labels_validate=test_labels
            )
        )

    # Baseline 4: Chance
    chance_level = [0.1 for i in n_samples_list]

    # Prepare plotting
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    label_fontsize = 16
    ticklabel_fontsize = label_fontsize
    title_fontsize = label_fontsize + 3 
    legend_fontsize = label_fontsize - 2
    ax1.set_ylabel ('Decoding accuracy', fontsize=label_fontsize)
    ax2.set_ylabel('Validation decoding accuracy', fontsize=label_fontsize)

    # Plot network performance
    architectures = {'net_1': '[2000-500-30]', 'net_2' : '[4000-2000-90]'}
    colors = {'net_1': 'tab:green', 'net_2': 'tab:cyan'}
    plot_labels = {'net_1': 'Original net size', 'net_2': 'Increased net size'}
    resultfolder = '../results/intermediate/'
    network_accuracy_validate_mean, network_accuracy_validate_std = load_network_accuracy(resultfolder, architectures, n_ipc_list, decode_mode='validate')
    network_accuracy_train_mean, network_accuracy_train_std = load_network_accuracy(resultfolder, architectures, n_ipc_list, decode_mode='train_classes')
    for networkIt in architectures:
        ax1.errorbar(n_samples_list, network_accuracy_train_mean[networkIt], network_accuracy_train_std[networkIt], label = plot_labels[networkIt], linewidth = 2.5, c = colors[networkIt])
        ax2.errorbar(n_samples_list, network_accuracy_validate_mean[networkIt], network_accuracy_validate_std[networkIt], label = plot_labels[networkIt], linewidth = 2.5, c = colors[networkIt])

    # Plot baselines
    ax1.plot(n_samples_list, sfa_decoding_accuracy_train, label='SFA inputs', linewidth=2, linestyle='-.', color='k')
    ax2.plot(n_samples_list, sfa_decoding_accuracy_validate, label='SFA inputs', linewidth=2, linestyle='-.', color='k')

    # ax1.plot(n_samples_list, kmeans_decoding_accuracy_train, label='k-means inputs', linewidth=2, linestyle ='--', color ='k')
    # ax2.plot(n_samples_list, kmeans_decoding_accuracy_validate, label='k-means inputs', linewidth=2, linestyle ='--', color ='k')
    
    ax1.plot(n_samples_list, linear_decoding_accuracy_train, label='LD inputs', linewidth=2, color='k', linestyle=':')
    ax2.plot(n_samples_list, linear_decoding_accuracy_validate, label='LD inputs', linewidth=2, color='k', linestyle=':')

    for ax in [ax1, ax2]:
        ax.plot(n_samples_list, chance_level, label='Chance level', c='black', linewidth=2)
        ax.legend(bbox_to_anchor =(1, 1), frameon=False, fontsize=legend_fontsize)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim([0, 1])
        ax.set_xlabel('Training sequences', fontsize=label_fontsize)
        ax.set_xticks(ticks=n_samples_list)
        ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)

    for fig in [fig1, fig2]:
        fig.tight_layout()

    # Save figures
    figures_path = f'../results/figures/fig5cd/'
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)
    fig1.savefig(figures_path + 'fig5c.svg')
    fig2.savefig(figures_path + 'fig5d.svg')
    plt.show()


