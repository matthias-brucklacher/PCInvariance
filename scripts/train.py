
"""Train a predictive coding network.

PyTorch implementation of a Hebbian Predictive Coding network described in Brucklacher, Bohte et al., 2022. 
Please cite: https://www.biorxiv.org/content/10.1101/2022.07.18.500392v2
Code author: Matthias Brucklacher (UvA Amsterdam).

"""
import argparse
from during_run_analysis.metrics import metrics
import json
from load_data import load_data, add_noise_to_data
from network import Network
import os
import torch
import numpy as np
import random

def main(args, train_data_original_format, validate_data_original_format, train_labels_original_format, validate_labels_original_format):
    args = args
    results_path = '../results/intermediate/' + args.resultfolder
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    else:    
        print("Directory " , results_path ,  " already exists")
    all_metrics = metrics(evaluate_on_validation_data=args.use_validation_data)
    if args.labels == 'smallnorb_labels.npy': assert args.use_validation_data == False, 'SmallNorb dataset does not contain enough instances for validation set'

    for runIt in range(args.n_runs):
        print('Starting run {} of {}'.format(runIt+1, args.n_runs))
        with torch.no_grad(): # No automated gradient calculation needed (learning is purely Hebbian)
            
            # Set hyperparameters for training 
            inference_steps = args.i_steps
            if args.do_train_static: 
                cycles_per_frame = args.cpf * 6 # Longer inference rate to compensate for staticity
                cycles_per_sequence = args.cps
            else:
                cycles_per_frame = args.cpf
                cycles_per_sequence = args.cps

            # Set up and initialize network, put on GPU
            size_list = args.architecture[:] 
            n_neurons_input_area = train_data_original_format.shape[-1]
            size_list.insert(0, n_neurons_input_area)
            net = Network(size_list, i_rate1=args.i_rates[0], i_rate2=args.i_rates[1], i_rate3=args.i_rates[2], device=device, l_rate=args.l_rate)
            minibatch_size = train_data_original_format.shape[0] # Different classes in parallel
            if flags['train_batchwise']:
                e0_batch, e1_batch, x1_batch, y1_batch, e2_batch, x2_batch, y2_batch, x3_batch, y3_batch = net.init_batches(minibatch_size) 
            else: 
                e0, e1, x1, y1, e2, x2, y2, x3, y3 = net.init_act(size_list) 

            # Load pretrained weights if desired. If multiple runs are done to estimate uncertainty, multiple weight sets need to be provided
            if flags["is_pretrained"] == True:
                net.load_weights(f'pretrained_weights/pretrained_net-{runIt}/')

            # Initialize evaluation metrics: Decoding accuracy and reconstruction error
            all_metrics.add_run()
            all_metrics.compute_and_append(results_path, flags['eval_steps'], runIt, net, train_data_original_format, train_labels_original_format, validate_data_original_format, validate_labels_original_format)

            # # Test for contesting of review 1 decision: reset more or less every three sequences
            rng = np.random.default_rng(12345)
            reset_list = []
            for i in range(args.epochs):
                lst = []
                for j in range(10):
                    a = rng.random()
                    if a < 0.3:
                        lst.append(1)
                    else:
                        lst.append(0)
                reset_list.append(lst)

            fixed_random_frameorder = [3, 0, 2, 4, 1, 5] # For shuffling control (supplementary)
            changing_random_frameorder = fixed_random_frameorder

            # Iterate through epochs
            for i in range(args.epochs):
                print("\nEpoch " + str(i+1) + " / " + str(args.epochs))
                e0_batch, e1_batch, x1_batch, y1_batch, e2_batch, x2_batch, y2_batch, x3_batch, y3_batch = net.init_batches(minibatch_size) # Reset network
                random.shuffle(changing_random_frameorder) # For shuffling control
                if flags['train_batchwise']:
                    for minibatchItDim1 in range(train_data_original_format.shape[1]): # Number of instances
                        for minibatchItDim2 in range(train_data_original_format.shape[2]): # Number of transformations
                            e0_batch, e1_batch, x1_batch, y1_batch, e2_batch, x2_batch, y2_batch, x3_batch, y3_batch = net.init_batches(minibatch_size) # Reset network
                            for k in range(cycles_per_sequence):
                                for frameIt in range(train_data_original_format.shape[3]):
                                    minibatch_frames= torch.transpose(train_data_original_format[:,minibatchItDim1,minibatchItDim2,frameIt,...],0,1)
                                    if args.do_train_static: # Reset network each frame, but grant extra inference time for fair comparison
                                        e0_batch, e1_batch, x1_batch, y1_batch, e2_batch, x2_batch, y2_batch, x3_batch, y3_batch = net.init_batches(minibatch_size) # Reset network
                                        e0_batch, e1_batch, x1_batch, y1_batch, e2_batch, x2_batch, y2_batch, x3_batch, y3_batch = net(minibatch_frames, cycles_per_frame*inference_steps, e0_batch, e1_batch, x1_batch, y1_batch, e2_batch, x2_batch, y2_batch, x3_batch, y3_batch)
                                    
                                    if args.noise_on:
                                        minibatch_frames = add_noise_to_data(minibatch_frames, device)
                                    for m in range(cycles_per_frame):
                                        e0_batch, e1_batch, x1_batch, y1_batch, e2_batch, x2_batch, y2_batch, x3_batch, y3_batch = net(minibatch_frames, inference_steps, e0_batch, e1_batch, x1_batch, y1_batch, e2_batch, x2_batch, y2_batch, x3_batch, y3_batch)
                                        for n in range(minibatch_size):
                                            net.learn(e0_batch[:,n], e1_batch[:,n], e2_batch[:,n], y1_batch[:,n], y2_batch[:,n], y3_batch[:,n])
                
                else: # Sequential training
                    e0, e1, x1, y1, e2, x2, y2, x3, y3 = net.init_act(size_list) 
                    # Iterate through classes, transformations, instances from classes and lastly frames
                    for classIt in range(train_data_original_format.shape[0]):
                        for trafoIt in range(train_data_original_format.shape[1]):
                            for instanceIt in range(train_data_original_format.shape[2]):
                                #if reset_list[i][instanceIt] == 1: # Reset only at given points
                                e0, e1, x1, y1, e2, x2, y2, x3, y3 = net.init_act(size_list) # Reset activity before each sequence
                                for k in range(cycles_per_sequence):
                                    for frameIt in range(train_data_original_format.shape[3]):
                                        frame = train_data_original_format[classIt,trafoIt,instanceIt,frameIt]
                                        if args.noise_on:
                                            frame = add_noise_to_data(frame, device)
                                        if args.do_train_static: # Reset each frame
                                            e0, e1, x1, y1, e2, x2, y2, x3, y3 = net.init_act(size_list) # Reset activity before each frame
                                            e0, e1, x1, y1, e2, x2, y2, x3, y3 = net(frame, cycles_per_frame*inference_steps, e0, e1, x1, y1, e2, x2, y2, x3, y3)
                                        for m in range(cycles_per_frame):
                                            e0, e1, x1, y1, e2, x2, y2, x3, y3 = net(frame, inference_steps, e0, e1, x1, y1, e2, x2, y2, x3, y3) # Inference steps
                                            net.learn(e0, e1, e2, y1, y2, y3) # Learning step
                print("Training completed for this epoch. Proceeding to evaluation")

                # Now evaluate model performance: 1. reconstruction error, 2. decoding accuracy on both training and validation subset
                all_metrics.compute_and_append(results_path, flags['eval_steps'], runIt, net, train_data_original_format, train_labels_original_format, validate_data_original_format, validate_labels_original_format)
        
        # Save weights
        dir_name = f'{results_path}/weights_run-{runIt}' 
        net.save_weights(dir_name)
        print('\nRun '+ str(runIt+1) + ' complete')

    # Average across runs and save evaluation metrics
    sim_id = args.data[:-4] + '_noise-' + str(args.noise_on) + '_static-' + str(args.do_train_static) # Identifier with the most important information to be stored in file names
    metric_mean_std_dict = all_metrics.mean_std()
    all_metrics.save(results_path, metric_mean_std_dict, sim_id)

    # Store used parameters
    with open(results_path + "/settings.txt", "w") as f:   
        f.write('Used data\n')
        f.write(args.data)
        f.write('\n of dimensions ' + str(train_data_original_format.shape))
        f.write('\n\nUsed arguments from parser\n')
        json.dump(args.__dict__, f, indent=2)
        f.write('\n\nUsed control flags \n')
        json.dump(flags, f, indent = 2)

if __name__ == '__main__':
    torch.manual_seed(0) # For reproducibility
    random.seed(0)
    if torch.cuda.is_available(): # Use GPU if possible
        dev = "cuda:0"
        print("Cuda is available")
    else:
        dev = "cpu"
        print("Cuda not available")
    global device
    device = torch.device(dev)

    # Settings for simulation run
    parser = argparse.ArgumentParser(description ='Start')
    parser.add_argument('--data', type = str, nargs = '?', help = 'input data file')
    parser.add_argument('--labels', type = str, nargs = '?', help = 'input label file')
    parser.add_argument('--resultfolder', type = str, nargs = '?', help = 'name of result folder')
    parser.add_argument('--architecture', type = int, nargs = 3, default= [2000, 500, 30], help = 'Layer sizes above input layer')
    parser.add_argument('--noise_on', type = int, nargs = '?', default = 0, help = 'Add noisy backgrounds')
    parser.add_argument('--do_train_static', type = int, nargs = '?', default = 0, help = 'Reset after every frame')
    parser.add_argument('--use_validation_data', type = int, nargs = '?', default = 0, help = 'Test network on validation split (only for MNIST digits).')
    parser.add_argument('--i_rates', type = float, nargs = 3, default= [0.05, 0.05, 0.05], help = 'i_rate_1 i_rate_2 i_rate_3')
    parser.add_argument('--l_rate', type = float, nargs = '?', default= 0.01, help = 'learning rate')
    parser.add_argument('--epochs', type = int, nargs = '?', default= 10, help = 'Number of epochs to run')
    parser.add_argument('--n_runs', type = int, nargs = '?', default= 1, help = 'Number of runs to compute statistics. Typically 1, 3 or 4.')
    parser.add_argument('--i_steps', type = int, nargs = '?', default = 10, help = 'Inference steps before weight update')
    parser.add_argument('--cps', type = int, nargs = '?', default = 5, help = 'Cycles per sequence')
    parser.add_argument('--cpf', type = int, nargs = '?', default = 10, help = 'Cycles per frame')
    parser.add_argument('--trafos', type = int, nargs = 3, default = [0, 0, 0], help = 'List of transformations')
    parser.add_argument('--n_instances_per_class_train', type = int, nargs = '?', default = 1, help = 'Number of instances per class in training data')

    args = parser.parse_args()

    # If launching the script from the IDE (as opposed to from the terminal), set arguments:
    if args.data is None:
        args = parser.parse_args(['--data', 'mnist_extended_fast.npy', '--labels','labels_mnist_extended.npy','--resultfolder',
                                        'results','--architecture', '2000', '500', '30', '--noise_on', '0', '--use_validation_data', '1',
                                        '--do_train_static', '0', '--epochs', '1', '--trafos', '1', '1', '1', '--n_instances_per_class_train', '1']) 

    # Set additional flags and parameters
    global flags
    flags = {
        "is_pretrained": False, # Load existing weights
        "eval_steps": 2000, # Inference steps used during testing (decoding and RDM computation),
        "train_batchwise": True # Whether to train in parallel
        }

    train_data_original_format, validate_data_original_format, train_labels_original_format, validate_labels_original_format = load_data(args.data, args.n_instances_per_class_train, args.trafos, args.labels, device)
    main(args, train_data_original_format, validate_data_original_format, train_labels_original_format, validate_labels_original_format)




