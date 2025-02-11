% Clear the workspace and initialize consistent random values
clc;
clear;
close all;

% Load the monitor data (add name to the directory)
load("net_results\gru_1L_96_0.020.mat");

% Set characteristics for plotting
line_width = 2;   
font_size = 12;
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

monitor_data = net_data.monitor_data;

% Find the first occurrence of zero in rmse_train
zero_idx = find(monitor_data.rmse_train == 0, 1);

% If there is a zero in rmse_train, cut the data
if ~isempty(zero_idx)
    monitor_data.rmse_train = monitor_data.rmse_train(1:zero_idx-1);
    monitor_data.rmse_train_smooth = monitor_data.rmse_train_smooth(1:zero_idx-1);
    monitor_data.rmse_validation = monitor_data.rmse_validation(1:zero_idx-1);
    monitor_data.iterations_store = monitor_data.iterations_store(1:zero_idx-1);
end

% Interpolate RMSE smooth and validation values
valid_smooth_idx = monitor_data.rmse_train_smooth > 0;
valid_validation_idx = monitor_data.rmse_validation > 0;

interp_rmse_train_smooth = interp1(monitor_data.iterations_store(valid_smooth_idx), monitor_data.rmse_train_smooth(valid_smooth_idx), monitor_data.iterations_store, 'linear', 'extrap');
interp_rmse_validation = interp1(monitor_data.iterations_store(valid_validation_idx), monitor_data.rmse_validation(valid_validation_idx), monitor_data.iterations_store, 'linear', 'extrap');

% Save the interpolated data back to monitor_data
monitor_data.rmse_train_smooth = interp_rmse_train_smooth;
monitor_data.rmse_validation = interp_rmse_validation;

% Plot RMSE metrics
F = figure;
plot(monitor_data.iterations_store, monitor_data.rmse_train, 'c', 'DisplayName', 'Training RMSE', 'LineWidth', line_width);
hold on;
plot(monitor_data.iterations_store, interp_rmse_train_smooth, 'b', 'DisplayName', 'Training RMSE Smooth', 'LineWidth', line_width);
plot(monitor_data.iterations_store, interp_rmse_validation, 'r', 'DisplayName', 'Validation RMSE', 'LineWidth', line_width);
xlabel('Iterations', 'FontSize', font_size, 'Interpreter', 'latex');
ylabel('RMSE values', 'FontSize', font_size, 'Interpreter', 'latex');
legend('Training RMSE','Training RMSE smoothed','Validation RMSE', 'Location', 'best', 'FontSize', font_size, 'Interpreter', 'latex');
title('RMSE Metrics', 'FontSize', font_size, 'Interpreter', 'latex');
xlim([min(monitor_data.iterations_store), max(monitor_data.iterations_store)]);
grid on;

% Plot a vertical dotted line at the lowest validation score iteration
xline(monitor_data.min_val_iteration, '--k', 'LineWidth', line_width, 'DisplayName',"Min evaluation rmse"); % Assuming all seeds are equal

hold off;

xlabel('Iterations', 'FontSize', font_size, 'Interpreter', 'latex');
legend('show', 'Location', 'best', 'FontSize', font_size, 'Interpreter', 'latex');
xlim([min(monitor_data.iterations_store), max(monitor_data.iterations_store)]);
grid on;

linkaxes(findall(gcf,'Type','axes'), 'x');
F.Color = 'w';