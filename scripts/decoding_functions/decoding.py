
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans  
from sklearn.metrics import accuracy_score
from sksfa import SFA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import linear_model

def linear_decodability(rep_array, label_array, n_splits):
    """Train and evaluate a linear decoder in stratified 3-fold manner, mapping a subset of a list of representations to correct labels.

    Args:
        rep_array (numpy.ndarray): Representations to decode. Shape is (n_classes * n_instances_per_class * n_trafos, n_frames_per_sequence, n_neurons). 
        label_array (numpy.ndarray): Array with data labels. Shape is (n_classes, n_instances_per_class, n_trafos, n_frames_per_sequence).

    Returns:
        linear_decoding_accuracy_train (float): Cross-validated decoding accuracy

    """
    # Prepare representations and labels
    if not isinstance(rep_array, np.ndarray):
        rep_array = rep_array.cpu().numpy()
    if not isinstance(label_array, np.ndarray):
        label_array = label_array.cpu().numpy()

    n_classes, n_instances_per_class, n_trafos, n_frames_per_sequence = label_array.shape[0], label_array.shape[1], label_array.shape[2], label_array.shape[3]
    label_array = label_array.reshape(n_classes * n_instances_per_class * n_trafos, n_frames_per_sequence)

    # Prepare representations and labels
    assert (rep_array.shape[0], rep_array.shape[1]) == (label_array.shape[0], label_array.shape[1]), "Dimension of labels must match dimension of representations"

    # Split dataset into train and test
    skf = StratifiedKFold(n_splits=n_splits) 
    skf.get_n_splits(rep_array, label_array)

    # Count number of unique labels and convert integer labels to one-hot vectors
    n_labels = np.unique(label_array, return_counts = False).shape[0]
    labels_onehot = labels_to_onehot(label_array)

    # Reshape label array. skl.split takes only 1-dimensional inputs -> Merge first two dimensions
    label_array = label_array.reshape(label_array.shape[0]*label_array.shape[1])
    labels_onehot = labels_onehot.reshape(labels_onehot.shape[0]*labels_onehot.shape[1], n_labels)

    # Reshape representation array. skl.split takes only 2-dimensional inputs.
    n_neurons = rep_array.shape[2]
    rep_array = rep_array.reshape(n_classes * n_instances_per_class * n_trafos * n_frames_per_sequence, n_neurons)

    # Compute accuracy of linear decoder for each train-test split.
    cumulative_accuracy = 0
    for train_index, test_index in skf.split(rep_array, label_array):
        reps_train, reps_test = rep_array[train_index], rep_array[test_index]
        labels_train, labels_test = labels_onehot[train_index], labels_onehot[test_index]
        reg = linear_model.LinearRegression()
        reg.fit(reps_train, labels_train)
        labels_predicted = reg.predict(reps_test)
        
        # Convert one-hot back to predicted label and calculate accuracy on test set: put test set into model, calculate fraction of TP+TN over all responses
        labels_predicted = (labels_predicted == labels_predicted.max(axis=1, keepdims=True)).astype(int)
        accuracy = accuracy_score(labels_test, labels_predicted)
        cumulative_accuracy += accuracy
    return cumulative_accuracy/n_splits

def test_generalization(reps_train, labels_train, reps_validate, labels_validate):
    """Train a linear classifier on all representations and labels from training data and evaluate its accuracy on the validation data.
    
    Args:
        reps_train(numpy.ndarray): Representations to train decoder on. Shape is (n_classes * n_instances_per_class * n_trafos, n_frames_per_sequence, n_neurons). 
        labels_train (numpy.ndarray): Array with data labels. Shape is (n_classes, n_instances_per_class, n_trafos, n_frames_per_sequence).
        reps_validate (numpy.ndarray): Representations inferred on unseen data. Shape is (n_classes * n_instances_per_class_validate * n_trafos, n_frames_per_sequence, n_neurons)
        labels_validate (numpy.ndarray): Labels of unseen data. Shape is (n_classes, n_instances_per_class_validate, n_trafos, n_frames_per_sequence).

    Returns:
        acc_validate (float): Accuracy on validation data.

    """
    assert(reps_train.shape[1:] == reps_validate.shape[1:]), 'Training and validation representations must have the same number of frames per sequence and neurons, but have' + \
        f'{reps_train.shape[1:]} and {reps_validate.shape[1:]}'

    # Reshape label arrays
    n_frames_per_sequence = reps_train.shape[1]
    labels_train = labels_train.reshape(-1, n_frames_per_sequence) # -> shape (n_classes * n_instances_per_class * n_trafos, n_frames_per_sequence)
    labels_validate = labels_validate.reshape(-1, n_frames_per_sequence) # -> shape (n_classes * n_instances_per_class * n_trafos, n_frames_per_sequence)

    # Convert last dim of both label arrays to one-hot-vectors
    labels_validate_onehot = labels_to_onehot(labels_validate)
    labels_train_onehot = labels_to_onehot(labels_train)

    # Flatten first two dimensions
    n_neurons = reps_train.shape[2]
    n_labels = labels_validate_onehot.shape[2] # Number of unique labels
    reps_train = reps_train.reshape(-1, n_neurons)
    reps_validate = reps_validate.reshape(-1, n_neurons)
    labels_validate_onehot = labels_validate_onehot.reshape(-1, n_labels)
    labels_train_onehot = labels_train_onehot.reshape(-1, n_labels)

    # Fit classifier on training representations and labels
    reg = linear_model.LinearRegression()
    reg.fit(reps_train, labels_train_onehot)
    labels_predicted = reg.predict(reps_validate)
    
    # Convert one-hot back to predicted label and calculate accuracy on test set: put test set into model, calculate fraction of TP+TN over all responses
    labels_predicted = (labels_predicted == labels_predicted.max(axis=1, keepdims=True)).astype(int)
    acc_validate = accuracy_score(labels_validate_onehot, labels_predicted)
    return acc_validate

def labels_to_onehot(label_array):
    """Convert numpy array of labels into one-hot vectors.

    Args:
        label_array (numpy.ndarray): numpy array of dimensions [n_classes, X] where X is the total number of frames per class

    Returns: 
        labels_onehot (numpy.ndarray): shape (n_classes, X, n_labels) where n_labels is the number of unique labels

    """
    n_labels = np.unique(label_array, return_counts=False).shape[0] # Number of unique labels

    # Initialize array
    labels_onehot = np.zeros([label_array.shape[0], label_array.shape[1], n_labels])

    for i in range(label_array.shape[0]):
        for j in range(label_array.shape[1]):
            labels_onehot[i, j, int(label_array[i, j])] = 1
    return labels_onehot

def kmeans_decoding(train_array, train_labels, n_clusters, validate_array=None, validate_labels=None):
    """K-Means clustering for readout. Decoding accuracy is calculated for the optimal assignment of the found clusters to the labels.

    Args:
        train_array (numpy.ndarray): Input patterns to be decoded. Shape can be (n_classes, n_instances, n_trafos, n_frames_per_sequence, n_neurons) 
            or (n_classes * n_instances * n_trafos * n_frames_per_sequence, width, height).
        validate_array (numpy.ndarray, optional): Validation set of input patterns. If none is provided, no validation accuracy is calculated.
            Shape can be (n_classes, n_instances, n_trafos, n_frames_per_sequence, n_neurons) 
            or (n_classes * n_instances * n_trafos * n_frames_per_sequence, width, height).
        train_labels (numpy.ndarray): Data labels. Shape is (n_classes, 1, n_trafos, n_instances)
        validate_labels (numpy.ndarray, optional): Data labels of validation set. Shape is (n_classes, 1, n_trafos, n_instances). Only needs to be provided if validation is desired. 

    Returns:
        accuracy_training (float): Accuracy on training data.
        accuracy_validation (float or None): Accuracy on validation data. None if no validation data was provided.

    """
    # Make sure data is in the right format
    train_array = make_input_data_2d(train_array) 
    validate_array = make_input_data_2d(validate_array)
    train_labels_list = train_labels.flatten().tolist()
    if validate_labels is not None:
        validate_labels_list = validate_labels.flatten().tolist()

    # Fit k-means clustering with given number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(train_array)
    cluster_assignments = kmeans.labels_

    # Go through all possible cluster assigments brute force and find the one that yields the highest readout accuracy
    decoding_list_unpermuted = []
    train_accuracies= []
    for i in range(n_clusters):
        decoding_list_unpermuted.append(i)
    k = 0
    readout_lists = [] 
    for i in itertools.permutations(decoding_list_unpermuted):
        k+=1
        if k % 100000 == 0: print(f'{k}/3.628.800 (for 10 classes)' ) # Status of this slow computation
        cluster_assignments_decoded = []
        for j in range(len(cluster_assignments)):
            cluster_assignments_decoded.append(i[cluster_assignments[j]]) # Replace labels according to this readout
        train_accuracies.append(accuracy_score(cluster_assignments_decoded, train_labels_list)) # Calculate accuracy for this readout
        readout_lists.append(i)
    accuracy_training = max(train_accuracies) # Choose best-fitting readout

    # Now calculate validation accuracy if desired (i.e. when validate_array is given)
    if validate_array is not None:
        validate_cluster_assignments = kmeans.predict(validate_array)
        validate_predictions = []
        best_readout_list = readout_lists[train_accuracies.index(max(train_accuracies))]
        for i in range(validate_array.shape[0]):
            validate_predictions.append(best_readout_list[validate_cluster_assignments[i]])
        accuracy_validation = accuracy_score(validate_predictions, validate_labels_list)
    else: 
        accuracy_validation = None
    return accuracy_training, accuracy_validation

def sfa_decoding(train_array, train_labels, n_components, validate_array=None, validate_labels=None):
    """
    Slow Feature Analysis followed by linear readout.

    Args:
        train_array (numpy.ndarray): Input patterns to be decoded. Shape is (n_classes, n_instances_per_class, n_trafos, n_frames_per_sequence, n_neurons) 
        validate_array (numpy.ndarray, optional): Validation set of input patterns. If none is provided, no validation accuracy is calculated.
            Shape is (n_classes, n_instances, n_trafos, n_frames_per_sequence, n_neurons).
        train_labels (numpy.ndarray): Data labels. Shape is (n_classes, 1, n_trafos, n_instances_per_class)
        validate_labels (numpy.ndarray, optional): Data labels of validation set. Shape is (n_classes, 1, n_trafos, n_instances_perr_class). Only needs to be provided if validation is desired. 

    Returns:
        accuracy_training (float): Accuracy on training data.
        accuracy_validation (float or None): Accuracy on validation data. Returns 'None' if no validation data was provided.
        
    """
    n_classes = train_array.shape[0]
    n_instances_per_class = train_array.shape[1]

    # Reshape data
    train_array = make_input_data_2d(train_array) 
    validate_array = make_input_data_2d(validate_array)

    # Extract slow features
    sfa = SFA(n_components=n_components)
    sfa.fit(train_array)
    extracted_features_training = sfa.transform(train_array)
    extracted_features_training = extracted_features_training.reshape(n_classes*n_instances_per_class, 6, -1) # From [n_classes * n_frames_per_sequence, n_components]-> [n_classes, n_frames_per_sequence, n_components]
    
    # Decode the extracted features 
    accuracy_training = linear_decodability(extracted_features_training, train_labels, n_splits=3)
    if validate_array is None:
        accuracy_validation = None
    else:
        extracted_features_validation = sfa.transform(validate_array.reshape(-1, validate_array.shape[-1]))
        extracted_features_validation = extracted_features_validation.reshape(-1, 6, n_components)
        accuracy_validation = test_generalization(extracted_features_training, train_labels, extracted_features_validation, validate_labels)
    return accuracy_training, accuracy_validation

def make_input_data_2d(data_array):
    """Flattens the first three and the last two dimensions of a data array. 
    
    Args:
        data_array (numpy.ndarray or torch.Tensor): Data array. Shape is (n_classes, n_instances, n_trafos, n_frames, width, height) 

    Returns:
        data_array_made_2d (numpy.ndarray or torch.Tensor): Reshaped data array. Shape is (n_classes * n_instances * n_trafos * n_frames, width * hight) 
    
    """
    if data_array is None: # Don't reshape invalid arrays
        return None
    n_frames = data_array.shape[0] * data_array.shape[1] * data_array.shape[2] * data_array.shape[3]
    n_neurons = data_array.shape[-1] * data_array.shape[-2]
    return data_array.reshape((n_frames, n_neurons))