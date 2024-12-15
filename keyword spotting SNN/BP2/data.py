from scipy.io import loadmat 
import numpy as np
import torch
def load_data(args):
    if args.dataset == "TIDIGITS":
        m = loadmat(args.dataset_path)
        max_time = args.max_time #ms
        input_neuron_num = args.n_input
        input_comb_coef = 10
        train_data_num = m['train_pattern'].shape[0]
        train_datasets = np.zeros((train_data_num, max_time, int(input_neuron_num/1)))
        for i in range(train_data_num):
            current_array = m['train_pattern'][i][0]
            current_spikes_input = np.zeros((max_time, int(input_neuron_num/1)))
            for input_idx in range(current_array.shape[0]):
                for spike_time in current_array[input_idx]:
                    if spike_time < max_time:
                        current_spikes_input[int(spike_time)][int(input_idx/1)] = 1
            train_datasets[i] = current_spikes_input
        train_datasets = torch.Tensor(train_datasets).byte()
        train_labels = torch.LongTensor(m['train_labels']-1)

        test_data_num = m['test_pattern'].shape[0]
        test_datasets = np.zeros((test_data_num, max_time, input_neuron_num))
        for i in range(train_data_num):
            current_array = m['test_pattern'][i][0]
            current_spikes_input = np.zeros((max_time, input_neuron_num))
            for input_idx in range(current_array.shape[0]):
                for spike_time in current_array[input_idx]:
                    if spike_time < max_time:
                        current_spikes_input[int(spike_time)][input_idx] = 1
            test_datasets[i] = current_spikes_input
        test_datasets = torch.Tensor(test_datasets).byte()
        test_labels = torch.LongTensor(m['test_labels']-1)

        train_dataset = torch.utils.data.TensorDataset(train_datasets, train_labels)
        test_dataset = torch.utils.data.TensorDataset(test_datasets,test_labels)

    return train_dataset, test_dataset