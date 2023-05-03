"""Compute raw autocorrelation values using a trained network.

"""
import numpy as np
import json
import torch
from scripts.network import Network
from scripts.load_data import load_data

def main():
    if torch.cuda.is_available(): # Use GPU if possible
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    with torch.no_grad():
        # Load trained network and data
        PATH_TO_WEIGHTS = '../results/intermediate/fig6/weights_run-0/'
        net = Network(size_list=[28*28, 2000, 500, 30], i_rate1=0.05, i_rate2=0.05, i_rate3=0.05, device=device, l_rate=0.00)
        net.load_weights(PATH_TO_WEIGHTS)
        train_data, _, _, _ = load_data(dataset='autocorrelation_mnist.npy', 
                                        labels_filename='labels_autocorrelation_mnist.npy', 
                                        n_instances_per_class_train=1,
                                        trafos=[0, 0, 0], 
                                        device=device)
        
        # Compute autocorrelations for a given list of populations
        populations = ['e0', 'e1', 'e2','y1', 'y2', 'y3']
        corr_per_lag, corr_per_lag_and_seq = autocorrelation(net, train_data, populations)
    
    # Store intermediate results
    AUTOCORR_RESULTFOLDER = '../results/intermediate/fig6/'
    dict_of_nparrays_to_json(corr_per_lag, AUTOCORR_RESULTFOLDER + 'corr_per_lag_autocorrelation_mnist.json')
    dict_of_nparrays_to_json(corr_per_lag_and_seq, AUTOCORR_RESULTFOLDER + 'corr_per_lag_and_seq_autocorrelation_mnist.json')

def autocorrelation(net, data, populations):
    """Compute autocorrelations of neural activity.

    Args:
        net (Network): Trained predictive coding network.
        data (): Input data (moving sequences). Shape is (n_classes, n_instances_per_class, n_trafos, n_frames_per_sequence, n_neurons).
        populations (list): List of populations in net for which to compute autocorrelations.

    Returns:
        corr_per_lag (dict): Correlation per time lag, averaged across sequences.
        corr_per_lag_and_sequence (dict): Correlation per time lag and per sequence.

    """
    datapoints_per_frame = 100 # how many timelags per frame to evaluate
    datapoints_per_seq = datapoints_per_frame * data.shape[3]  
    max_lag = datapoints_per_seq - 1
    number_of_sequences = data.shape[0]
    n_int = 10 # integrate activity over this amount of timesteps. Each datapoint then contains n_int inference steps
    reps = dict.fromkeys(populations)
    average_act = dict.fromkeys(populations)
    corr = dict.fromkeys(populations)
    size_dict = {'e0': net.size_list[0], 
                 'e1': net.size_list[1], 
                 'e2': net.size_list[2], 
                 'y0': net.size_list[0], 
                 'y1': net.size_list[1], 
                 'y2':net.size_list[2], 
                 'y3': net.size_list[3]}
    
    # Initialize some arrays
    for popIt in populations:
        reps[popIt] = np.zeros([number_of_sequences, data.shape[3] * datapoints_per_frame, size_dict[popIt]])
        corr[popIt] = np.zeros([number_of_sequences, max_lag, size_dict[popIt]])

    # Compute representations for autocorrelation
    print("Computing representations for autocorrelation")
    for i in range(number_of_sequences): 
        e0, e1, x1, y1, e2, x2, y2, x3, y3 = net.init_act(net.size_list) # Reset network
        e0, e1, x1, y1, e2, x2, y2, x3, y3 = net(data[i, 0, 0, 0], 3000, e0, e1, x1, y1, e2, x2, y2, x3, y3) 
        for j in range(data.shape[3]): # Iterate through all frames of the sequence
            frame = data[i, 0, 0, j]
            # Now average neural activity over multiple timesteps
            for k in range(datapoints_per_frame):
                for popIt in populations:
                    average_act[popIt] = torch.zeros(size_dict[popIt]).to(net.device)
                for l in range(n_int):
                    e0, e1, x1, y1, e2, x2, y2, x3, y3   = net(frame, 1, e0, e1, x1, y1, e2, x2, y2, x3, y3) # Single inference step
                    average_act['e0'] += torch.clone(e0) / n_int # integrate activity
                    average_act['e1'] += torch.clone(e1) / n_int 
                    average_act['y1'] += torch.clone(y1) / n_int 
                    average_act['e2'] += torch.clone(e2) / n_int
                    average_act['y2'] += torch.clone(y2) / n_int 
                    average_act['y3'] += torch.clone(y3) / n_int
                for popIt in populations:
                    reps[popIt][i, datapoints_per_frame * j + k] = average_act[popIt].cpu().numpy()
                    
    print("Representations computed. Computing autocorrelations.")

    # Now we have a list of representations. Iterate through it and compute autocorrelation for a given lag.
    for popIt in populations:
        for lag in range(max_lag): 
            a = np.multiply(reps[popIt][:, :datapoints_per_seq - lag], reps[popIt][:, lag:datapoints_per_seq])
            corr[popIt][:, lag] = np.mean(a, axis=(1)) # Average across times
    
    # Now, compute more specialized metrics
    corr_per_lag_and_sequence = dict.fromkeys(populations)
    corr_per_lag = dict.fromkeys(populations)

    for popIt in populations:
        # corr[popIt] has three axis: sequence, lag, neuron 
        corr_per_lag_and_sequence[popIt] = np.average(corr[popIt], axis=(2)) # Average over neurons in subpopulation
        for seqIt in range(number_of_sequences): #data.shape[0]):
            corr_per_lag_and_sequence[popIt][seqIt] /= corr_per_lag_and_sequence[popIt][seqIt, 0] # Normalize

        # For illustration purposes compute the same, but specific for individual lags
        corr_per_lag[popIt] = np.average(corr[popIt], axis=(0,2)) # Average over sequence and neurons in subpopulations
        corr_per_lag[popIt] /= corr_per_lag[popIt][0] # Normalize

    print('Autocorrelations computed')
    return corr_per_lag, corr_per_lag_and_sequence

def dict_of_nparrays_to_json(dictionary, filename):
    """Store dictionary of numpy arrays in json file.

    Args:
        dictionary (dict): Dictionary of numpy.ndarrays.
        filename (str): Name under which to store the file.

    """
    a_file = open(filename, 'w')
    json.dump({k: v.tolist() for k, v in dictionary.items()}, a_file)
    a_file.close()

if __name__=='__main__':
    main()

