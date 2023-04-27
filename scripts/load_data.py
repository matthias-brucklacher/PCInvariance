import torch
import numpy as np

""" 
Functions to load data and labels as well as to add noise to them.
"""

def load_data(dataset, n_instances_per_class_train, trafos, labels_filename, device):

    PATH_TO_DATA = f'../data/data_preprocessed/{dataset}'
    data = torch.tensor(np.load(PATH_TO_DATA), device=device)
    print("Data " + dataset + " loaded successfully")

    selected_trafos = list(set(trafos)) # Keep only unique values #[0, 1, 2] # List to specify which transformations are included: 0 - translation, 1 - rotation, 2 - scaling
    if labels_filename in ['smallnorb_labels.npy', 'labels_autocorrelation_mnist.npy']: # These datasets do not contain instances per class to create a validation dataset. 
        n_instances_per_class_validate = 0 
        print('X')
    else: 
        n_instances_per_class_validate = 20 # Don't change (standardized validation set)
    print(str(n_instances_per_class_train) + ' instances per class in training data')
    print(str(n_instances_per_class_validate) + ' instances per class in validation data')
    n_instances_selected = n_instances_per_class_train + n_instances_per_class_validate # Training + validation

    assert all(trafo in list(range(data.shape[2])) for trafo in selected_trafos), 'Selected transformations must be present in dataset' 
    assert n_instances_selected <= data.shape[1], "Number of instances selected ({}) must be smaller than total number of instances in dataset ({})".format(n_instances_selected, data.shape[1])
    data = data[:, :, selected_trafos]
    data = torch.flatten(data, start_dim=-2)  # Flatten last two dimensions (image width and height). Shape is now [n_classes, n_trafos, n_instances, height*width]

    PATH_TO_LABELS = f'../data/data_preprocessed/{labels_filename}'
    labels = np.load(PATH_TO_LABELS)
    labels = labels[:, :, selected_trafos]

    assert data.shape[0:3] == labels.shape[0:3], "Dimension of labels must match dimension of data"

    # Split into training and validation subsets.
    validate_labels = labels[:, -n_instances_per_class_validate:, :] # take last 20 samples
    validate_data = data[:, -n_instances_per_class_validate:, :]
    train_labels = labels[:, :n_instances_per_class_train, :]
    train_data = data[:, :n_instances_per_class_train, :]

    return train_data, validate_data, train_labels, validate_labels

def add_noise(frame, device):
    """Takes input image and fills empty parts with random noise.

    """
    background = torch.normal(-0.1, 0.5, [frame.shape[0]], device=device) # Weak background, managable by the network
    background = torch.clamp(background, min=0, max=1) # Keep only positive values
    noisy_frame = torch.where(frame == 0, background, frame) # Place figure in front of noise: Apply noise only where figure is not
    return noisy_frame

def add_noise_to_data(data, device):
    """Takes input data and fills empty parts with random noise. Returns noisy_data of same dimensions.

    """
    background = torch.normal(-0.1, 0.5, data.shape, device = device) # Weak background, managable by the network
    background = torch.clamp(background, min = 0, max = 1) # Keep only positive values
    noisy_data = torch.where(data == 0, background, data) # Place figure in front of noise: Apply noise only where figure is not
    return noisy_data