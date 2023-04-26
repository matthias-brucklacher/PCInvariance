import torch
import numpy as np

""" 
Functions to load data and labels as well as to add noise to them.
"""

def load_data(dataset, n_instances_per_class_train, trafos, labels_filename, device):

    data_path = f'../data/data_preprocessed/{dataset}'
    data_original_format = torch.tensor(np.load(data_path), device = device) # (samples, sequencelength, dim_x, dim_y)
    print("Data " + dataset + " loaded successfully")
    data_width = data_original_format.shape[-1]

    selected_trafos = list(set(trafos)) # Keep only unique values #[0,1,2] # List to specify which transformations are included: 0 - translation, 1 - rotation, 2 - scaling
    if labels_filename != 'smallnorb_labels.npy': # Smallnorb dataset as used here does not contain instances per class to create a validation dataset. 
        n_instances_per_class_validate = 20 # Don't change (standardized validation set)
    else: n_instances_per_class_validate = 0 
    print(str(n_instances_per_class_train) + ' instances per class in training data')
    print(str(n_instances_per_class_validate)+ ' instances per class in validation data')
    n_instances_selected = n_instances_per_class_train + n_instances_per_class_validate # Training + validation

    assert all(trafo in list(range(data_original_format.shape[2])) for trafo in selected_trafos), 'Selected transformations must be present in dataset' 
    assert n_instances_selected <= data_original_format.shape[1], "Number of instances selected ({}) must be smaller than total number of instances in dataset ({})".format(n_instances_selected, data_original_format.shape[1])
    data_original_format = data_original_format[:,:,selected_trafos]
    data_original_format = torch.flatten(data_original_format, start_dim = -2)  # Flatten last two dimensions (image width and height). Shape is now [n_classes, n_trafos, n_instances, height*width]
    data = torch.flatten(data_original_format, end_dim = -3)

    labels_path = f'../data/data_preprocessed/{labels_filename}'
    data_labels_original_format = np.load(labels_path)
    data_labels_original_format = data_labels_original_format[:,:,selected_trafos]

    assert data_original_format.shape[0:3] == data_labels_original_format.shape[0:3], "Dimension of labels must match dimension of data"

    """
    Split into training and validation subsets
    """

    # Two views of each dataset (useful later). In the "_original_format", the dimensions of n_classes, n_trafos, n_instances are maintained
    validate_labels_original_format = data_labels_original_format[:,-n_instances_per_class_validate:,:] # take last 20 samples
    validate_data_original_format = data_original_format[:,-n_instances_per_class_validate:, :]
    train_labels_original_format = data_labels_original_format[:, :n_instances_per_class_train, :]
    train_data_original_format = data_original_format[:, :n_instances_per_class_train, :]

    # Historic dataset of rotating digits. In the current manuscript version, this is only used for the autocorrelation results, and requires pretrained weights.
    load_old_data = False
    if load_old_data:
        train_data_original_format = torch.tensor(np.load('rotData.npy'), device = device)
        data_width = train_data_original_format.shape[-1]
        train_data_original_format = torch.flatten(train_data_original_format, start_dim = -2)
        train_data_original_format = train_data_original_format[:, None, None, :, :]
        validate_data_original_format = train_data_original_format
        validate_labels_original_format = train_labels_original_format
    return train_data_original_format, validate_data_original_format, train_labels_original_format, validate_labels_original_format

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