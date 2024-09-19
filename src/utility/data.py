import numpy as np

def format_hierarchical_data_me(train_dict, valid_dict, test_dict, num_bins):
    train_event_bins = make_times_hierarchical(train_dict['T'].cpu().numpy(), num_bins=num_bins)
    valid_event_bins = make_times_hierarchical(valid_dict['T'].cpu().numpy(), num_bins=num_bins)
    test_event_bins = make_times_hierarchical(test_dict['T'].cpu().numpy(), num_bins=num_bins)
    train_events = train_dict['E'].cpu().numpy()
    valid_events = valid_dict['E'].cpu().numpy()
    test_events = test_dict['E'].cpu().numpy()
    train_data = [train_dict['X'].cpu().numpy(), train_event_bins, train_events]
    valid_data = [valid_dict['X'].cpu().numpy(), valid_event_bins, valid_events]
    test_data = [test_dict['X'].cpu().numpy(), test_event_bins, test_events]
    return train_data, valid_data, test_data

def calculate_layer_size_hierarch(layer_size, n_time_bins):
    def find_factors(n):
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                factor1 = i
                factor2 = n // i
                if factor1 < factor2:
                    return factor1, factor2
        return (1, n_time_bins)
    result = find_factors(n_time_bins)
    return [(layer_size, result[0]), (layer_size, result[1])]
    
def make_times_hierarchical(event_times, num_bins):
    min_time = np.min(event_times[event_times != -1]) 
    max_time = np.max(event_times[event_times != -1]) 
    time_range = max_time - min_time
    bin_size = time_range / num_bins
    binned_event_time = np.floor((event_times - min_time) / bin_size)
    binned_event_time[binned_event_time == num_bins] = num_bins - 1
    return binned_event_time