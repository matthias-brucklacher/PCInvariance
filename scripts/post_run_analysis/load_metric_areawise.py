"""Helper function to load area-wise metrics, such as reconstruction errors.

"""
import pickle
def load_metric_areawise(filename):
    """Takes metrics that was recorded areawise, but stored in a single file and returns the recorded values per area.

    See the metric class for which metrics are stored in this manner.
    
    Example:
        file_path = 'simulation_id/metrics/rec_errors_mnist_extended_fast_noise-0_static-0'
        reconstruction_errors_mean_area_1, reconstruction_errors_mean_area_1, reconstruction_errors_mean_area_3 = load_metric_areawise(file_path)

    Args:
        filename (str): Path to metric file consisting of an epochwise list of 3-tuples (one float per area).  
    
    Returns: 
        data_area_1 (list of floats): Metric over epochs area 1
        data_area_2 (list of floats): Metric over epochs area 2
        data_area_3 (list of floats): Metric over epochs area 3

    """
    with open(filename +'.txt', "rb") as fp:
        metric_data = pickle.load(fp)
        data_area_1 = []
        data_area_2 = []
        data_area_3 = []
        for i in range(len(metric_data)):
            data_area_1.append(metric_data[i][0])
            data_area_2.append(metric_data[i][1])
            data_area_3.append(metric_data[i][2])
    return data_area_1, data_area_2, data_area_3