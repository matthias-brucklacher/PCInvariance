""" Plot reconstruction errors after network training.

Example (from terminal): 
    $ python train.py --data mnist_extended_fast.npy --labels labels_mnist_extended.npy --trafos 0 0 0 --resultfolder fig7b_translation --epochs 20 --n_runs 4
    $ python train.py --data mnist_extended_fast.npy --labels labels_mnist_extended.npy --trafos 1 1 1 --resultfolder fig7b_rotation --epochs 20 --n_runs 4
    $ python train.py --data mnist_extended_fast.npy --labels labels_mnist_extended.npy --trafos 2 2 2 --resultfolder fig7b_scaling --epochs 20 --n_runs 4
    
    $ python post_run_analysis/plot_fig7b.py --simulation_id_1 fig7b_translation --simulation_id_2 fig7b_rotation --simulation_id_3 fig7b_scaling

"""

import argparse
from load_metric_areawise import load_metric_areawise
import matplotlib.pyplot as plt
import os

def plot_reconstruction_errors_areawise():
    """Plot reconstruction errors after network training.

    """
    parser = argparse.ArgumentParser(description ='Plot reconstruction error over epochs')
    parser.add_argument('--simulation_id_1', type = str, nargs = '?', help = 'Directory with "metric" files of first condition (transformation) to plot.')
    parser.add_argument('--simulation_id_2', type = str, nargs = '?', help = 'Directory with "metric" files of second condition.')
    parser.add_argument('--simulation_id_3', type = str, nargs = '?', help = 'Directory with "metric" files of third condition.')
    args = parser.parse_args()

    # Gather necessary data: Mean and standard deviation.
    err1_means, err2_means, err3_means = [], [], []
    err1_std, err2_std, err3_std = [], [], []

    for simulation_id in [args.simulation_id_1, args.simulation_id_2, args.simulation_id_3]:
        intermediate_results_path = '../results/intermediate/' + simulation_id

        # Means
        err1_data, err2_data, err3_data = load_metric_areawise(intermediate_results_path + '/metrics/rec_errors_mnist_extended_fast_noise-0_static-0')
        err1_means.append(err1_data)
        err2_means.append(err2_data)
        err3_means.append(err3_data)

        # Standard deviations
        err1_std_data, err2_std_data, err3_std_data = load_metric_areawise(intermediate_results_path + '/metrics/rec_errors_std_mnist_extended_fast_noise-0_static-0')
        err1_std.append(err1_std_data)
        err2_std.append(err2_std_data)
        err3_std.append(err3_std_data)

    # Plotting
    label_fontsize = 16
    title_fontsize = label_fontsize +3
    legend_fontsize = label_fontsize - 2

    fig, axs = plt.subplots(1, 3, sharey=True)
    axs[0].set_ylabel('Reconstruction MSE', fontsize=label_fontsize)
    axs[1].set_xlabel('Training epoch', fontsize=label_fontsize)

    area = 1
    err_means = [err1_means, err2_means, err3_means]
    err_std= [err1_std, err2_std, err3_std]
    dataset_labels = ['Fast translation', 'Fast rotation', 'Scaling', 'Fast translation static', 'Fast rotation static', 'Scaling static']
    colors = ['tab:green','tab:orange','tab:blue', 'k']
    for ax in axs:
        for i in range(3):
            ax.errorbar(range(21), err_means[area-1][i], err_std[area-1][i], label=dataset_labels[i], color=colors[i])
        ax.set_ylim(bottom=0)
        ax.set_title('Area ' + str(area), fontsize=title_fontsize)
        area += 1
        ax.set_box_aspect(1.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    axs[2].legend(loc=(1.1,0.4),prop={'size': legend_fontsize}, frameon=False)
    plt.gcf().subplots_adjust(left=0.15, right=0.7)

    # Keep ticklabels only in central and left plot
    plt.setp(axs[1].get_yticklabels(), visible=False)
    plt.setp(axs[2].get_yticklabels(), visible=False)
    plt.setp(axs[0].get_xticklabels(), visible=False)
    plt.setp(axs[2].get_xticklabels(), visible=False)

    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    
    # Save plot
    figures_path = '../results/figures/fig7b'
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)
    plt.tight_layout()
    plt.savefig(figures_path + '/reconstruction_errors_epochwise.svg')
    
if __name__ == '__main__':
    plot_reconstruction_errors_areawise()