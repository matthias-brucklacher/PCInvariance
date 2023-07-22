"""
This script measures the capacity to reconstruct full inputs, and plots the reconstructions.

"""
from scripts.load_data import load_data
from matplotlib import pyplot as plt
from scripts.network import Network
import numpy as np


# Set up network
img_width = 34
size_list = [img_width ** 2, 2000, 500, 30]
i_rates = [0.05, 0.05, 0.05]
l_rate = 0.01
device = 'cpu'
net = Network(size_list, i_rate1=i_rates[0], i_rate2=i_rates[1], i_rate3=i_rates[2], device=device, l_rate=l_rate)
# For rotation
net.load_weights(f'../results/intermediate/fig4b/weights_run-0/')
dataset, _ , _, _ = load_data(dataset='mnist_extended.npy', n_instances_per_class_train=1, trafos=[1], labels_filename='labels_mnist_extended.npy', device=device)
net.to(device)

n_classes = dataset.shape[0]
reconstruct_classes = [i for i in range(n_classes)]
reconstruct_frames = [1, 5]
for class_it in reconstruct_classes:
    for frame_it in reconstruct_frames:
        # with noise
        #net.reconstruct_topdown(add_noise(train_data[i[0], i[1]], device), flags["eval_steps"], save_as = args.resultfolder+ '/reconstruct_topdown_'+ str(i[0])+'-'+str(i[1])+'.svg')   
        frame = dataset[class_it, 0, 0, frame_it]# (n_pixels, 1)     
        net.reconstruct_topdown(frame, i_steps=2000, save_as = '../results/figures/fig7a/reconstruct_topdown_'+ str(class_it)+'-'+str(frame_it)+'.svg') 
