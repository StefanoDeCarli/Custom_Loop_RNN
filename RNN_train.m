% Trains a RNN network, either LSTM or GRU. Provides a network with specified characteristics, at the minimum validation RMSE.
function [net,info,monitor,net_name] = RNN_train(train_dataset, valid_dataset, train_options)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   PARAMETERS EXPLANATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   train_dataset   [struct]        Structured dataset for training, with x and y as trials, already normalized
%   valid_dataset   [struct]        Structured dataset for validation, with x and y as trials, already normalized
%   train_options   [struct]        Contains the training options below
%   - is_lstm       [logical]       Select LSTM (true) or GRU (false) architecture
%   - hidden_units  [N x 1 double]  Number of hidden unit, implicit number of layers N
%   - learn_rate    [double]        Learn rate, static
%   - lasso_lambda  [double]        Weight for the Lasso regularization
%   - pruning_th    [double]        Threshold to prune the weights below
%   - epochs_pruned [double]        Interval between prunings
%   - max_epochs    [double]        Maximum number of epochs to train with
%   - mini_batch    [double]        Number of trials used in training for each iteration
%   - dropout_rate  [double]        Dropout rate, part of the network to be dropped at each iteration
%   - is_visible    [logical]       Whether the training monitor is shown or not
%   - is_verbose    [logical]       Whether loss updates are shown or not in command window

% The dataset structures must be like:

% dataset = struct(...
%     'description', "General structure to store a dataset", ... % add a description
%     'x', [], ...       % x data, dimension being [N_trial x 1 cell] , each trial being [N_steps x N_inputs double]
%     'y', [], ...       % y data, dimension being [N_trial x 1 cell] , each trial being [N_steps x N_outputs double]
%     'x_mean', [], ...  % mean of x data; not necessary; not necessary, recommended
%     'y_mean', [], ...  % mean of y data; not necessary; not necessary, recommended
%     'x_std', [], ...   % standard deviation of x data; not necessary, recommended
%     'y_std', [] ...    % standard deviation of y data; not necessary, recommended
% );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   UNPACK OPTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

is_lstm = train_options.is_lstm;
hidden_units = train_options.hidden_units;
learn_rate = train_options.learn_rate;
lasso_lambda = train_options.lasso_lambda;
pruning_th = train_options.pruning_th;
epochs_pruned = train_options.epochs_pruned;
max_epochs = train_options.max_epochs;
mini_batch = train_options.mini_batch;
dropout_rate = train_options.dropout_rate;
is_visible = train_options.is_visible;
is_verbose = train_options.is_verbose;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   SETTING PARAMETERS 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% General parameters to set, CAN BE CHANGED

validation_frequency = 5e-3;    % Relative number of validation check during training
window_RMSE = 5;                % Select on how many interactions to print a smoothed RMSE training value

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   DATA SCRAPING & PREALLOCATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Utility functions directory
addpath(genpath([pwd, filesep, 'utilities']));

% Scrape and initialize data
num_layers = height(hidden_units);
num_features = width(valid_dataset.x{1});
num_responses = width(valid_dataset.y{1});

network_layers = net_maker(is_lstm, hidden_units, num_features, num_responses, dropout_rate);

net_name = generate_net_name(is_lstm, num_layers, hidden_units, learn_rate);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   PRE-INITIALIZE LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create network from layers
net = dlnetwork(network_layers);

% Initialize batches
[x_train_batch, y_train_batch] = create_mini_batches(train_dataset.x, train_dataset.y, mini_batch);

% Initialize monitor data
num_iterations_per_epoch = size(x_train_batch,1);
num_iterations = num_iterations_per_epoch * max_epochs;
validation_steps = ceil(validation_frequency * num_iterations);

% Select custom loss function
custom_loss = @standard_clf;

% If different functions use the following code instead
% if is_lstm
%     custom_loss = @clf_lstm;
% else
%     custom_loss = @clf_gru;
% end

% Initialize training monitor
monitor = generate_monitor(is_visible);

% Initialize monitor related data
window_data = zeros(window_RMSE,1);
loss_data = zeros(num_iterations,1);

monitor.Progress = 0;

% Initialize iteration and epoch counters
iteration = 0;
epoch = 0;

% Initialize data for adam solver
average_grad = [];
average_sqgrad = [];

% Select best execution environment, nice if GPU available
executionEnvironment = "auto";
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    updateInfo(monitor, ExecutionEnvironment="GPU");
    net = dlupdate(@gpuArray, net);
else
    updateInfo(monitor, ExecutionEnvironment="CPU");
end

% Preprocess batches
[dlx, dly] = preprocess_mini_batches(x_train_batch, y_train_batch);

% Preallocate arrays to store metrics and iterations
max_iterations = max_epochs * num_iterations_per_epoch;
rmse_train = zeros(max_iterations, 1);
rmse_train_smooth = zeros(max_iterations, 1);
rmse_validation = zeros(max_iterations, 1);
iterations_store = zeros(max_iterations, 1);

% Initialize min validation RMSE and corresponding network, to later save
% min val network in the end
min_validation_rmse = inf;
min_val_net = [];
min_info = struct(...
    'training_rmse', [], ...
    'validation_rmse', []);
min_monitor_data = struct(...
    'rmse_train', [], ...
    'rmse_train_smooth', [], ...
    'rmse_validation', [], ...
    'iterations_store', []);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   CUSTOM LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

monitor_data_index = 0;  % Index for storing data in monitor_data

while epoch < max_epochs && ~monitor.Stop
    epoch = epoch + 1;
    % Print the epoch
    if is_verbose
    fprintf('%d. ', epoch);
    end

    % Pruning step: apply pruning every epochs_pruned
    if mod(epoch, epochs_pruned) == 0
        net = prune_weights(net, pruning_th);
    end

    % Shuffle and prepare batches for this epoch
    index = randperm(num_iterations_per_epoch);
    x_train_batch = dlx(index);
    y_train_batch = dly(index);
    batch = 0;

    while batch < num_iterations_per_epoch && ~monitor.Stop
        batch = batch + 1;
        iteration = iteration + 1;
        monitor_data_index = monitor_data_index + 1;
        net = resetState(net);

        % Evaluate loss function
        [losses, gradients, ~] = dlfeval(custom_loss, net, x_train_batch{batch}, y_train_batch{batch}, lasso_lambda, is_verbose);
        
        % Update network parameters based on loss
        [net, average_grad, average_sqgrad] = adamupdate(net, gradients, average_grad, average_sqgrad, iteration, learn_rate);

        current_rmse_train = double(sqrt(losses.mse_loss));
        rmse_train(monitor_data_index) = current_rmse_train;

        loss_data(iteration) = losses.mse_loss;

        if rem(iteration, window_RMSE) == 0 || iteration == 1
            if iteration == 1
                smooth_rmse = double(sqrt(losses.mse_loss));
                recordMetrics(monitor, iteration, TrainingRMSE_smooth=smooth_rmse);
            else
                for window_index = 0:(window_RMSE-1)
                    window_data(window_index + 1) = loss_data(iteration - window_index);
                end
                smooth_rmse = double(sqrt(mean(window_data)));
                recordMetrics(monitor, iteration, TrainingRMSE_smooth=smooth_rmse);
            end
            % Store training smooth RMSE and iteration
            rmse_train_smooth(monitor_data_index) = smooth_rmse;
        end

        if rem(iteration, validation_steps) == 0 || iteration == 1
            validation_rmse = rmse_validator(net, valid_dataset.x, valid_dataset.y);
            recordMetrics(monitor, iteration, ValidationRMSE=double(validation_rmse));
            % Store validation RMSE and iteration
            rmse_validation(monitor_data_index) = double(validation_rmse);
            iterations_store(monitor_data_index) = iteration;

            % Save min_val_net if validation_rmse is minimal
            if validation_rmse < min_validation_rmse
                min_validation_rmse = validation_rmse;
                min_val_net = net;

                % Save the relevant information for min_val_net
                min_info.training_rmse = double(extractdata(sqrt(losses.mse_loss)));
                min_info.validation_rmse = validation_rmse;

                min_monitor_data.rmse_train = rmse_train(1:monitor_data_index);
                min_monitor_data.rmse_train_smooth = rmse_train_smooth(1:monitor_data_index);
                min_monitor_data.rmse_validation = rmse_validation(1:monitor_data_index);
                min_monitor_data.iterations_store = iterations_store(1:monitor_data_index);

                % Save the iteration when min_val_net is triggered
                recorded_monitor.min_val_iteration = iteration;
            end
        end

        iterations_store(monitor_data_index) = iteration;
        recordMetrics(monitor, iteration, TrainingRMSE=current_rmse_train);

        updateInfo(monitor, Epoch=(string(epoch) + " of " + string(max_epochs)), Iteration=(string(iteration) + " of " + string(num_iterations)), LearnRate=learn_rate);
        monitor.Progress = 100 * iteration/num_iterations;
    end
end

% Save all data in monitor
recorded_monitor.rmse_train = rmse_train;                % Save RMSE for training
recorded_monitor.rmse_train_smooth = rmse_train_smooth;  % Save smoothed training RMSE
recorded_monitor.rmse_validation = rmse_validation;      % Save validation RMSE
recorded_monitor.iterations_store = iterations_store;    % Save the iterations

% Prune the network before saving
net = prune_weights(net, pruning_th);

net = min_val_net;
info = min_info;
monitor = recorded_monitor;
end