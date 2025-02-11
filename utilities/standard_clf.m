% Custom loss function for RNN networks
function [losses, gradients, state] = standard_clf(net, x, targets, lambda, is_verbose)

% Forward pass through the network
[Y, state] = forward(net, x);

% Calculate squared differences
squared_diff = (Y - targets).^2;

% Compute mean over all elements
losses.mse_loss = sum(squared_diff, 'all') / numel(Y);

% Compute lasso loss regularization term
losses.lasso_loss = lambda * sum(cellfun(@(w) sum(abs(w), 'all'), net.Learnables.Value)) / numel(net.Learnables.Value);

loss = losses.mse_loss + losses.lasso_loss;

% Compute gradients of the loss with respect to the learnable parameters
gradients = dlgradient(loss, net.Learnables);

% Print the loss
if is_verbose
    fprintf('RMSE = %.4f; Lasso = %.4e;\n', losses.mse_loss^0.5, losses.lasso_loss);
end
end