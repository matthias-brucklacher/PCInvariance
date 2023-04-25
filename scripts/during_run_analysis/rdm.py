"""
Functions for RDM computation.

Example:
    compute_rdm(y2_reps_train, train_data_original_format.cpu().numpy().shape, results_path)

"""

import torch
import matplotlib.pyplot as plt
import numpy as np

def distance(vec_a, vec_b):
    """Computes the cosine distance between two vectors.

    Args:
        vec_a (numpy.ndarray): First vector.
        vec_b (numpy.ndarray): Second vector.

    Returns:
        cosine_distance (float): Cosine distance between the two vectors.

    """
    assert vec_a.shape == vec_b.shape, "Vectors must have equal dimension"
    distance = 1-np.dot(vec_a, vec_b)/(np.linalg.norm(vec_a)*np.linalg.norm(vec_b))
    return distance

def torch_distance(a, b):
    """Computes the cosine distance between two vectors.

    Args:
        vec_a (torch.Tensor): First vector.
        vec_b (torch.Tensor): Second vector.

    Returns:
        cosine_distance (torch.Tensor): Cosine distance between the two vectors.

    """
    assert a.shape == b.shape, "Vectors must have equal dimension"
    distance = 1-torch.dot(a, b)/(torch.linalg.norm(a)*torch.linalg.norm(b))
    return distance

def compute_rdm(rep_array, dataset_shape, resultfolder):
        """Compute and plot all pairwise dissimilarity-valeus from a list of representations in a RDM (representational dissimilarity matrix).

        Args:
            rep_array (numpy.ndarray): Representations to be compared. Shape is (N_sequences, N_frames_per_sequence, N_neurons)
            dataset_shape (tuple): Dataset dimensions (n_classes, n_instances_per_class, n_trafos, n_frames_per_sequence, width*height)

        Returns:
            rdm (numpy.ndarray): RDM as 2D array.
            rdm_quality_classwise (float): Scalar measure of RDM invariance within each class.
            rdm_quality_instancewise (float): Scalar measure of RDM invariance within different views of the same instance.
        """
        reps = rep_array.reshape(-1, rep_array.shape[-1])

        rdm = np.zeros([reps.shape[0], reps.shape[0]])
        for i in range(reps.shape[0]):
            for j in range(reps.shape[0]): 
                rdm[i, j] = distance(reps[i], reps[j])
        
        # Compute measures of RDM quality: Low values within blocks, and high values outside blocks are good.
        rdm_quality_classwise = 0 
        rdm_quality_instancewise = 0

        n_classes = dataset_shape[0]
        n_instances_per_class = dataset_shape[1]
        n_instances = n_classes*n_instances_per_class
        n_trafos = dataset_shape[2]
        n_frames_per_sequence = dataset_shape[3]
        n_frames_per_instance = n_trafos*n_frames_per_sequence
        n_frames_per_class = n_instances_per_class*n_frames_per_instance

        for i in range(n_classes):
            start_idx = i*n_frames_per_class
            end_idx = (i+1)*n_frames_per_class
            class_within = np.average(rdm[start_idx:end_idx,start_idx:end_idx])
            class_to_outside = np.average(rdm[start_idx:end_idx,:])- class_within
            rdm_quality_classwise += (class_to_outside - class_within)
        rdm_quality_classwise/= n_classes # normalize

        for i in range(n_instances): # instance-wise
            start_idx = i*n_frames_per_instance
            end_idx = (i+1)*n_frames_per_instance
            instance_within = np.average(rdm[start_idx:end_idx,start_idx:end_idx])
            instance_to_outside = np.average(rdm[start_idx:end_idx,:])- instance_within
            rdm_quality_instancewise += (instance_to_outside - instance_within)

        rdm_quality_instancewise/= n_instances # normalize

        np.save(resultfolder+"/rdm.npy", rdm)

        # Plot RDM and save figure
        fig, ax = plt.subplots()
        size = 18
        im = ax.matshow(rdm, vmin = 0) # vmax = 1)
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('Sample #', size=size)
        ax.set_ylabel('Sample #', size=size)
        ax.tick_params(axis='both', which='major', labelsize=size)
        cbar_ax = fig.add_axes([0.73, 0.17, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax).set_label(label='Cosine Dissimilarity', size=size)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gcf().subplots_adjust(right=0.75)
        plt.savefig(resultfolder+ "/rdm.svg")
        plt.close()

        return rdm, rdm_quality_classwise, rdm_quality_instancewise



