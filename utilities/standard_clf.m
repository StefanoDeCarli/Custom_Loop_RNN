% Custom loss function for RNN networks
function [loss, gradients, state] = standard_clf(net, x, targets, lambda, is_verbose)

% Forward pass through the network
[Y, state] = forward(net, x);

% Calculate squared differences
squared_diff = (Y - targets).^2;

% Compute mean over all elements
mse_loss = sum(squared_diff, 'all') / numel(Y);

% Compute lasso loss regularization term
lasso_loss = lambda * sum(cellfun(@(w) sum(abs(w), 'all'), net.Learnables.Value)) / numel(net.Learnables.Value);

loss = mse_loss + lasso_loss;

% Compute gradients of the loss with respect to the learnable parameters
gradients = dlgradient(loss, net.Learnables);

% Print the loss
if is_verbose
    fprintf('RMSE = %.4f; Lasso = %.4e;\n', mse_loss^0.5, lasso_loss);
end
end