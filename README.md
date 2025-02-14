# Custom-Loop-RNN
Small Toolbox to for RNN custom loop in matlab (LSTM and GRU)

## Folder Structure

The toolbox consists of the following main components:

1. `RNN_train.m`: The primary function for training the ISS-promoted RNN.
2. `utilities`: A folder containing utility functions required for the training process.
3. `monitor_data_viewer.m`: A script for visualizing the training progress and ISS metrics.
4. `trainer.m`: A script to test the functionality of the training process using a provided dataset.
5. `data`: Contains the dataset for testing and validation purposes.
6. `README.md`: This file provides an overview and instructions for using the toolbox.

## RNN_train.m

This function trains an a RNN network, either LSTM or GRU, and returns the trained network, training information, monitor data, and the network name. The network is trained to achieve the minimum validation RMSE, which is returned.

### Function Signature

```matlab
function [net, info, monitor, net_name] = ISS_train(train_dataset, valid_dataset, ...
    is_lstm, hidden_units, dropout_rate, ...
    u_max_inputs, learn_rate, max_epochs, mini_batch, visible, penalty, epsilon)
```

### Parameters

-   `train_dataset`: Structured dataset for training, with `x` and `y` as trials, already normalized.
-   `valid_dataset`: Structured dataset for validation, with `x` and `y` as trials, already normalized.
-   `is_lstm`: Boolean flag to select LSTM (`true`) or GRU (`false`) architecture.
-   `hidden_units`: Array specifying the number of hidden units in each layer.
-   `dropout_rate`: Dropout rate for the network.
-   `u_max_inputs`: Array specifying the maximum values the inputs can assume.
-   `learn_rate`: Learning rate for training.
-   `max_epochs`: Maximum number of epochs for training.
-   `mini_batch`: Number of trials used in training for each iteration.
-   `visible`: Toggle on or off the monitor visibility.
-   `penalty`: Set strength of ISS promotion, recommended at 0.05 for start.
-   `visible`: Set margin of ISS promotion, recommended at 0.05 for start.

### Dataset Structure

The dataset should be a struct with the following fields:

-   `description`: Description of the dataset.
-   `x`: Cell array of input data, each cell containing a matrix of size `[N_steps x N_inputs]`.
-   `y`: Cell array of output data, each cell containing a matrix of size `[N_steps x N_outputs]`.

The data should not only contain the `train_dataset` and `valid_dataset`, but also a `stats` struct containing means and standard deviations of the inputs and outputs, in `[1 x N_inputs]` and `[1 x N_outputs]` format. The `stats` struct is shown here:
```matlab
stats = struct(...
    'description', "General structure to store a dataset", ...
    'u_mean', [], ...
    'u_std', [], ...
    'y_mean', [], ...
    'y_std', [], ...
);
```

utilities
-------------

This folder contains utility functions required for the training process. Ensure that this folder is included in your MATLAB path.

monitor_data_viewer.m
---------------------

This script visualizes the training progress and ISS metrics. It loads the monitor data from a saved network results file and generates plots for RMSE and ISS metrics.

trainer.m
--------

This script tests the functionality of the training process using the provided dataset `data.mat`. It demonstrates the ISS-promoted training.

Theory: strict Input-to-State Stability (ISS)
--------------------------------------

Strict Input-to-State Stability (ISS) is a desirable property in dynamic systems that ensures the system's state remains bounded in response to bounded inputs. This is crucial for ensuring realistic and stable behavior in RNNs, especially when modeling physical systems.

### Implementation in RNNs

To promote ISS in RNNs, we incorporate stability constraints into the training process. This involves:

-   **Custom Loss Function**: Adding a penalty term to the loss function that favors ISS.
-   **Regularization**: Using techniques like dropout and weight regularization to prevent overfitting and ensure stable learning.
-   **Architecture Design**: Carefully designing the network architecture, including the number of hidden units and layers, to balance complexity and stability.

Getting Started
---------------

1.  Clone or download the repository to your local machine. 
2.  Run the `tester.m` script to train a sample network and save the results.
3.  Use the `monitor_data_viewer.m` script to visualize the training progress and ISS metrics.
4.  Do whatever you want with the file :blush:

License
-------

This project is licensed under the MIT License. See the LICENSE file for more details.

Happy training with the ISS-Promoted RNN Toolbox!
