% Custom loss function for RNN networks
function [loss, gradients, state] = standard_clf(net, x, targets)

% Forward pass through the network
[Y, state] = forward(net, x);

% Calculate squared differences
squared_diff = (Y - targets).^2;

% Compute mean over all elements
loss = sum(squared_diff, 'all') / numel(Y);

% Compute gradients of the loss with respect to the learnable parameters
gradients = dlgradient(loss, net.Learnables);

% Print the loss
fprintf('RMSE = %.4f;\n', loss^0.5);
end