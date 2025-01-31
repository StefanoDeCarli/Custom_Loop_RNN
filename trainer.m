% Clear the workspace and initialize consistent random values
clc;
clear;
close all;
random_seed = 1;
rng(random_seed);

% Load dataset
data = load(fullfile('data', 'data.mat'));
varName = fieldnames(data);   % Get the field name(s) in the structure
data = data.(varName{1});     % Access the contents using dynamic field referencing

train_dataset = data.train_dataset;
valid_dataset = data.valid_dataset;

%% Select network hyperparameters

% Hidden units definition
N = 96;
H = floor(0.577 * N); % Two-layer: H ≈ 0.577N, similar amount of parameters to 1 layer with N hidden units
Z = floor(0.447 * N); % Three-layer: Z ≈ 0.447N, similar amount of parameters to 1 layer with N hidden units

% Architecture design
train_options.is_lstm = false;
train_options.hidden_units = [N]; % i.e. [Z;Z;Z] for 3 layers of Z hidden units each
train_options.dropout_rate = 0.05;

% Regularization design
train_options.lasso_lambda = 1e-5;
train_options.pruning_th = 5e-3;
train_options.epochs_pruned = 10;

% Training design
train_options.learn_rate = 2e-2;
train_options.max_epochs = 1000;
train_options.mini_batch = numel(train_dataset.x);    % Take all the trials, to change in case
train_options.is_visible = 'on';                      % on / off, show the training monitor or not

%% Train the network
[net,info,monitor,net_name] = RNN_train(train_dataset, valid_dataset, train_options);   % Training;

%% NET SAVE
% Initialize the net_results struct
net_data = struct(...
    'net', net, ...
    'info', info, ...
    'monitor_data', monitor, ...
    'stats', data.stats);

save(['net_results/', net_name], 'net_data');
clc;