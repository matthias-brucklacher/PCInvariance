"""Compute statistical analysis from autocorrelation metrics.

"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import scipy.stats

def analyze_autocorrelation():
    # First, load relevant data
    AUTOCORR_RESULTFOLDER = '../results/intermediate/fig6'

    data_name = 'autocorrelation_mnist'
    corr_per_lag = dict_of_nparrays_from_json(AUTOCORR_RESULTFOLDER + '/corr_per_lag_' + data_name + '.json')
    corr_per_lag_and_seq = dict_of_nparrays_from_json(AUTOCORR_RESULTFOLDER + '/corr_per_lag_and_seq_' + data_name+'.json')

    # Control flags
    fit_type = 'to_fixed' # Definition of time constant: From exp fit, or decay to fixed value

    populations = ['e0', 'e1', 'e2','y1', 'y2', 'y3']
    c_en =  ['black', 'grey', 'silver']
    c_rn = ['green', 'blue', 'red']
    c_all = [*c_en, *c_rn]
    label_dict_linebreaks = {'e0': 'ENs \n area 0', 'e1': 'ENs \n area 1', 'e2': 'ENs \n area 2', 'y1': 'RNs \n area 1', 'y2': 'RNs \n area 2', 'y3': 'RNs \n area 3'}
    label_dict = {'e0': 'ENs area 0', 'e1': 'ENs area 1', 'e2': 'ENs area 2', 'y1': 'RNs area 1', 'y2': 'RNs area 2', 'y3': 'RNs area 3'}

    # Fit function to estimate decay constant
    tau = dict.fromkeys(populations)
    tau_avg = dict.fromkeys(populations)
    tau_std = dict.fromkeys(populations)
    for popIt in populations:
        tau[popIt] = np.zeros(corr_per_lag_and_seq[popIt].shape[0])
        for seqIt in range(corr_per_lag_and_seq[popIt].shape[0]):
            xdata = np.arange(0, corr_per_lag_and_seq[popIt].shape[1])
            ydata = corr_per_lag_and_seq[popIt][seqIt]
            if fit_type == 'exp':
                popt, pcov = curve_fit(exp_function, xdata, ydata, bounds = ([0, 0.3, 0], [3., 10000., 1. ]))
                tau[popIt][seqIt] = popt[1]
            else:
                thr = 1 / np.exp(1)
                idx_below_thr = np.flatnonzero(ydata < thr) # Find all values below threshold
                # if does not fall below threshold, use linear approx
                if(idx_below_thr.size == 0): 
                    tau[popIt][seqIt] = 10*(thr-ydata[0])*(xdata[-1] - xdata[0])/(ydata[-1] - ydata[0]) 
                else:
                    tau[popIt][seqIt] = 10 * xdata[idx_below_thr[0]] # Take time of first value below threshold
        tau_avg[popIt] = np.mean(tau[popIt])
        tau_std[popIt] = np.std(tau[popIt])

    # Average values for ENs and RNs
    tau_en = (tau_avg['e0'] + tau_avg['e1'] + tau_avg['e1']) / 3
    tau_rn = (tau_avg['y1'] + tau_avg['y2'] + tau_avg['y3']) / 3
    print('Average time constant for ENs is ' + str(tau_en))
    print('Average time constant for RNs is ' + str(tau_rn))

    # Plot decay constants inferred from fit
    bar_thickness = 0.3
    fontsize = 16
    legend_fontsize = 16
    title_fontsize = fontsize + 3
    tau_values = list(tau_avg.values())
    tau_errors = list(tau_std.values())
    tau_labels = list(label_dict_linebreaks.values())
    tau_colors = c_all
    tau_colors.reverse()
    tau_values.reverse()
    tau_errors.reverse()
    tau_labels.reverse()
    ax = plt.subplot(111)
    ax.bar(tau_labels, tau_values, yerr=tau_errors, color=tau_colors, width=bar_thickness)
    ax.tick_params(axis='x', which='major', labelsize=fontsize)
    ax.set_ylabel("Decay constant $\\tau$ [time steps]", fontsize=fontsize)
    ax.set_title('Timescales of subpopulations', fontsize=title_fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()

    # Save plot
    FIGURES_RESULTFOLDER = '../results/figures/fig6'
    if not os.path.exists(FIGURES_RESULTFOLDER):
        os.mkdir(FIGURES_RESULTFOLDER)
    plt.savefig(FIGURES_RESULTFOLDER + '/timeconstants.svg')
    plt.close()

    # ANOVA table within ENs
    sequences_per_group = corr_per_lag_and_seq['e0'].shape[0] # How many sequences
    e_groups = ['e0', 'e1', 'e2']
    low_groups = ['e0', 'y1', 'e1']
    middle_groups = ['e1', 'y2', 'e2']
    high_groups = ['e2','y3']
    y_groups = ['y1', 'y2', 'y3']
    all_groups = populations

    anova(e_groups, tau_avg, tau, sequences_per_group)
    anova(y_groups, tau_avg, tau, sequences_per_group)
    anova(['y1', 'y2'], tau_avg, tau, sequences_per_group)
        
    # Plot decay of autocorrelation, first for RNs, then for ENs
    autocorr_xlabel = 'Time lag $\Delta$ [time steps]'
    autocorr_ylabel = 'Correlation $R$ with shifted activity'
    color_idx = 0
    fig, ax = plt.subplots()
    for popIt in ['y1', 'y2', 'y3']:
        plt.plot(range(0, corr_per_lag[popIt].shape[0] * 10, 10),
                  corr_per_lag[popIt],
                  label=label_dict[popIt],
                  color=c_rn[color_idx],
                  linewidth=2)
        color_idx += 1
    plt.legend(frameon=False, fontsize=legend_fontsize, loc='upper right')

    plt.title("Representation neurons", fontsize=title_fontsize)
    plt.xlabel(autocorr_xlabel, fontsize=fontsize)
    plt.ylabel(autocorr_ylabel, fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(FIGURES_RESULTFOLDER + '/autocorr_rep.svg')
    
    # Now for error neurons
    color_idx = 0
    fig, ax = plt.subplots()
    for popIt in ['e0','e1','e2']:
        plt.plot(range(0, 10 * corr_per_lag[popIt].shape[0], 10), 
                 corr_per_lag[popIt], 
                 label=label_dict[popIt], 
                 color=c_en[color_idx], 
                 linewidth=2)
        color_idx += 1
    plt.legend(frameon=False, fontsize=legend_fontsize)
    plt.title("Error neurons", fontsize=title_fontsize)
    plt.xlabel(autocorr_xlabel, fontsize=fontsize)
    plt.ylabel(autocorr_ylabel, fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(FIGURES_RESULTFOLDER + '/autocorr_err.svg')
    plt.show()
    plt.close()

def anova(groups, tau_avg, tau, sequences_per_group):
    """Perform analysis of variance (ANOVA), print the results.

    Args:
        groups (list): Each string in the list must be a neuronal population in tau and tau_avg.
        tau (dict): Time constants per population and sequence.
        tau_avg (dict): Time constants averaged across sequences.
        sequences_per_group (int): Number of sequences over which autocorrelation was computed.
    
    """
    k_groups = len(groups) # How many groups
    samplesize = k_groups * sequences_per_group

    # Calculate grand mean
    total = 0
    for popIt in groups:
        total += tau_avg[popIt]
    grand_mean = total / k_groups

    DFt = samplesize - 1 # Total degrees of freedom
    DFg = k_groups - 1
    DFe = DFt - DFg

    SSg = 0 # Variability across groups
    SSt = 0 # Total variability in response variable

    for groupIt in groups:
        SSg += sequences_per_group * (tau_avg[groupIt] - grand_mean) ** 2 
        SSt += np.sum((tau[groupIt] - grand_mean) ** 2)
    SSe = SSt - SSg # Variability within groups

    MSe = SSe / DFe # Average variability within groups
    MSg = SSg / DFg # Average variability across groups

    F_value = MSg / MSe 
    p_value = scipy.stats.f.sf(F_value, dfn=DFg, dfd=DFe)
    print('p value of populations ' + str(groups) + ' is ' + str(p_value))

    # If significant: Do multiple comparisons to see which pair is significantly different
    if p_value > 0.05: 
        print('ANOVA failed to reject null hypothesis')
    else:
        p_star = 0.05 / (k_groups - 1) # Bonferroni correction
        start_idx = 0
        for group1 in groups:
            start_idx += 1
            if start_idx < len(groups):
                for group2 in groups[start_idx:]:
                    Se = np.sqrt(MSe/sequences_per_group+MSe/sequences_per_group) # Standard error
                    DF = DFe
                    T_score = (tau_avg[group1] - tau_avg[group2]) / Se
                    p_value = 2*(1 - scipy.stats.t.cdf(abs(T_score), DF))
                    print('Pairwise comparison of ' + group1 + ' and ' + group2 + ' yielded a p-value of ' 
                          + str(p_value) + ' at diff. ' + str(np.abs(tau_avg[group1] - tau_avg[group2])))

def dict_of_nparrays_from_json(filename):
    """From json file back to dictionary.
    
    Args:
        filename (str): File location.
    
    Returns:
        content (dict): Original dictionary of numpy array from the json file.

    """
    a_file = open(filename, "r")
    content = a_file.read()
    content = json.loads(content)
    for popIt in content.keys():
        content[popIt] = np.asarray(content[popIt])
    return content

def exp_function(x, a, b, c):
    """Exponentially decaying function.

    """
    return a * np.exp(-x / b) + c

if __name__=='__main__':
    analyze_autocorrelation()
