"""The metric class used during network training to record metrics of interest such as the readout accuracy or reconstruction error.

Example:
    all_metrics = metrics(evaluate_on_validation_data=args.use_validation_data)
    all_metrics.add_run()
    all_metrics.compute_and_append(results_path, flags['eval_steps'], runIt, net, train_data_original_format, train_labels_original_format, validate_data_original_format, validate_labels_original_format)
    metric_mean_std_dict = all_metrics.mean_std()
    all_metrics.save(results_path, metric_mean_std_dict, sim_id)

"""
import numpy as np
import os
import pickle
from scripts.during_run_analysis.rdm import compute_rdm
from scripts.decoding_functions.decoding import linear_decodability, test_generalization

class metrics():
    def __init__(self, evaluate_on_validation_data):
        """Dictionary of metrics to log. Each value is a list of length n_runs containing each a list of length n_epochs.

        Args:
            evaluate_on_validation_data (bool): Whether or not the model shall be tested on unseen samples.

        """
        self.metrics = {'linear_decoding_accuracy_train_classes' : [],
                        'rec_err_area-1_train': [],
                        'rec_err_area-2_train': [],
                        'rec_err_area-3_train': [],
                        'rdm_quality_classwise': [],
                        'rdm_quality_instancewise_area-1': [],
                        'rdm_quality_instancewise_area-2': [],
                        'rdm_quality_instancewise_area-3': []
                        }
        
        # Add some more metrics for out-of-sample testing ('validation')
        self.evaluate_on_validation_data = evaluate_on_validation_data
        if evaluate_on_validation_data:            
            self.metrics.update({
                'linear_decoding_accuracy_validate': [],
                'rec_err_area-1_validate': [],
                'rec_err_area-2_validate': [],
                'rec_err_area-3_validate': [],
            }
            )

    def add_run(self):
        """Add empty list to metrics dictionary, to be filled during epochs of new run.

        """
        for keyIt in self.metrics:
            self.metrics[keyIt].append([])

    def compute_and_append(self, results_path, eval_steps, current_run, net, train_data_original_format, train_labels_original_format, validate_data_original_format=None, validate_labels_original_format=None):
        """Run network evaluation and append to metrics dictionary.

        Args:
            results_path (str): Directory in which to store intermediate results from the current run.
            eval_steps (int): Number of inference steps when inferring representations from the individual inputs.
            current_run (int): Number of current run.
            net (Network): Predictive coding network.
            train_data
            train_labels
            validate_data (optional)
            validate_labels (optional)
        
        """
        # Linear decoding class-wise on training data
        reconstruction_error1_train, reconstruction_error2_train, reconstruction_error3_train, y1_reps_train, y2_reps_train, y3_reps_train = net.reconstruction_error(train_data_original_format, eval_steps)
        linear_decoding_accuracy_train_classes = linear_decodability(y3_reps_train, train_labels_original_format, n_splits = 3)
        print(f'Area 3 decoding accuracy is {linear_decoding_accuracy_train_classes:.2%}')
        acc_area2 = linear_decodability(y2_reps_train, train_labels_original_format, n_splits = 3)
        acc_area1 = linear_decodability(y1_reps_train, train_labels_original_format, n_splits = 3)
        self.metrics['linear_decoding_accuracy_train_classes'][current_run].append(linear_decoding_accuracy_train_classes)

        # Reconstruction errors on training data
        self.metrics['rec_err_area-1_train'][current_run].append(reconstruction_error1_train)
        self.metrics['rec_err_area-2_train'][current_run].append(reconstruction_error2_train)
        self.metrics['rec_err_area-3_train'][current_run].append(reconstruction_error3_train)

        # Linear decoding class-wise on validation data   
        if self.evaluate_on_validation_data == True:
            assert (validate_data_original_format is not None) and (validate_labels_original_format is not None), 'Metrics initialized for validation but no validation data was given.'
            # This is too slow for local computation
            reconstruction_error1_validate, reconstruction_error2_validate, reconstruction_error3_validate, y1_reps_validate, y2_reps_validate, y3_reps_validate = net.reconstruction_error(validate_data_original_format, eval_steps)
            linear_decoding_accuracy_validate = test_generalization(y3_reps_train, labels_train=train_labels_original_format, reps_validate=y3_reps_validate, labels_validate=validate_labels_original_format)
            self.metrics['linear_decoding_accuracy_validate'][current_run].append(linear_decoding_accuracy_validate)
            print(f'Area 3 validation accuracy is {linear_decoding_accuracy_validate:.2%}')

            # Reconstruction errors on validation data
            self.metrics['rec_err_area-1_validate'][current_run].append(reconstruction_error1_validate)
            self.metrics['rec_err_area-2_validate'][current_run].append(reconstruction_error2_validate)
            self.metrics['rec_err_area-3_validate'][current_run].append(reconstruction_error3_validate)

        # RDM quality class- and instance-wise
        rdm2, rdm_quality_classwise2, rdm_quality_instancewise_area2 = compute_rdm(y2_reps_train, train_data_original_format.cpu().numpy().shape, results_path)
        rdm1, rdm_quality_classwise1, rdm_quality_instancewise_area1 = compute_rdm(y1_reps_train, train_data_original_format.cpu().numpy().shape, results_path)
        rdm, rdm_quality_classwise, rdm_quality_instancewise_area3 = compute_rdm(y3_reps_train, train_data_original_format.cpu().numpy().shape, results_path)
       
        self.metrics['rdm_quality_classwise'][current_run].append(rdm_quality_classwise)
        self.metrics['rdm_quality_instancewise_area-1'][current_run].append(rdm_quality_instancewise_area1)
        self.metrics['rdm_quality_instancewise_area-2'][current_run].append(rdm_quality_instancewise_area2)
        self.metrics['rdm_quality_instancewise_area-3'][current_run].append(rdm_quality_instancewise_area3)
    
    def mean_std(self):
        """Compute mean and standard deviation of metrics across multiple runs with different seeds.
        
        Returns:
            mean_std_dict (dict): Keys are metrics, values are lists of tuples (one for each epoch): (mean, std).

        """
        mean_std_dict  = dict.fromkeys(list(self.metrics.keys()))
        
        for keyIt in self.metrics:
            n_epochs = len(self.metrics[keyIt][0])
            mean_std_dict[keyIt] = np.zeros((n_epochs, 2)) 
            metrics = np.array(self.metrics[keyIt]) # convert to numpy array of shape (n_runs, n_epochs)

            # Now compute mean and stddev
            mean_std_dict[keyIt][:,0] = np.mean(metrics, axis=0).tolist()
            mean_std_dict[keyIt][:,1] = np.std(metrics, axis=0).tolist()
        return mean_std_dict
    
    def save_helper(self, results_path , metric_mean_std_dict, sim_id, key):
        """Saves mean and std of key-selected metric from metric_mean_std_dict.

        Args:
            results_path (str): Path to directory in which to store the mean_std_dic.
            metric_mean_std_dict (dict): Dictionary of metrics. Keys are metrics, values are lists of tuples (one for each epoch): (mean, std).
            sim_id (str): Identifier of simulation.
            key (str): The metric from metric_mean_std_dict that is selected for saving.

        """
        resultfolder = results_path + '/metrics'
        with open(resultfolder + "/" + key + "_" + sim_id + ".txt", "wb") as fp:   
            pickle.dump(metric_mean_std_dict[key][:, 0], fp)
        with open(resultfolder+ "/" + key + "_std_" + sim_id + ".txt", "wb") as fp:   
            pickle.dump(metric_mean_std_dict[key][:, 1], fp)

    def save(self, results_path, metric_mean_std_dict, sim_id):
        """Save dictionaries of list of floats.

        Args:
            results_path (str): Path to directory in which to store the mean_std_dic.
            metric_mean_std_dict (dict): Dictionary of metrics. Keys are metrics, values are lists of tuples (one for each epoch): (mean, std).
            sim_id (str): Identifier of simulation.

        """
        resultfolder = results_path + '/metrics'
        if not os.path.exists(resultfolder):
            os.mkdir(resultfolder)
        else:    
            print("Directory " , resultfolder,  " already exists")
        with open(resultfolder+ "/linear_decoding_accuracy_train_classes_" + sim_id + ".txt", "wb") as fp:   
            pickle.dump(metric_mean_std_dict['linear_decoding_accuracy_train_classes'][:, 0], fp)
        with open(resultfolder+ "/linear_decoding_accuracy_train_classes_std_" + sim_id + ".txt", "wb") as fp:   
            pickle.dump(metric_mean_std_dict['linear_decoding_accuracy_train_classes'][:, 1], fp)

        if self.evaluate_on_validation_data: 
            self.save_helper(results_path, metric_mean_std_dict, sim_id, 'linear_decoding_accuracy_validate')
        self.save_helper(results_path, metric_mean_std_dict, sim_id, 'rdm_quality_classwise')
        self.save_helper(results_path, metric_mean_std_dict, sim_id, 'rdm_quality_instancewise_area-3')

        # Combine reconstruction errors into tuples and save
        rec_errors_train = []
        rec_errors_train_std = []
        rdm_quality_layerwise = []
        rdm_quality_layerwise_std = []
        rec_errors_validate= []
        rec_errors_validate_std = []
        for i in range(len(metric_mean_std_dict['rec_err_area-1_train'])):
            rec_errors_train.append((metric_mean_std_dict['rec_err_area-1_train'][i][0], metric_mean_std_dict['rec_err_area-2_train'][i][0], metric_mean_std_dict['rec_err_area-3_train'][i][0]))
            rec_errors_train_std.append((metric_mean_std_dict['rec_err_area-1_train'][i][1], metric_mean_std_dict['rec_err_area-2_train'][i][1], metric_mean_std_dict['rec_err_area-3_train'][i][1]))
            rdm_quality_layerwise.append((metric_mean_std_dict['rdm_quality_instancewise_area-1'][i][0], metric_mean_std_dict['rdm_quality_instancewise_area-2'][i][0], metric_mean_std_dict['rdm_quality_instancewise_area-3'][i][0]))
            rdm_quality_layerwise_std.append((metric_mean_std_dict['rdm_quality_instancewise_area-1'][i][1], metric_mean_std_dict['rdm_quality_instancewise_area-2'][i][1], metric_mean_std_dict['rdm_quality_instancewise_area-3'][i][1]))
            
            if self.evaluate_on_validation_data:
                rec_errors_validate.append((metric_mean_std_dict['rec_err_area-1_validate'][i][0], metric_mean_std_dict['rec_err_area-2_validate'][i][0], metric_mean_std_dict['rec_err_area-3_validate'][i][0]))
                rec_errors_validate_std.append((metric_mean_std_dict['rec_err_area-1_validate'][i][1], metric_mean_std_dict['rec_err_area-2_validate'][i][1], metric_mean_std_dict['rec_err_area-3_validate'][i][1]))
        
        with open(resultfolder+ "/rec_errors_" + sim_id + ".txt", "wb") as fp:   
            pickle.dump(rec_errors_train, fp)
        with open(resultfolder+ "/rec_errors_std_" + sim_id + ".txt", "wb") as fp:   
            pickle.dump(rec_errors_train_std, fp)
        with open(resultfolder+ "/rdm_quality_layerwise_" + sim_id + ".txt", "wb") as fp:   
            pickle.dump(rdm_quality_layerwise, fp)
        with open(resultfolder+ "/rdm_quality_layerwise_std_" + sim_id + ".txt", "wb") as fp:   
            pickle.dump(rdm_quality_layerwise_std, fp)
        
        if self.evaluate_on_validation_data:
            with open(resultfolder+ "/rec_errors_validate_" + sim_id + ".txt", "wb") as fp:   
                pickle.dump(rec_errors_validate, fp)
            with open(resultfolder+ "/rec_errors_validate_std_" + sim_id + ".txt", "wb") as fp:   
                pickle.dump(rec_errors_validate_std, fp)
