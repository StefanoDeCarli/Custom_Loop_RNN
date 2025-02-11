function net = prune_weights(net, threshold)
    for i = 1:numel(net.Learnables.Value)
        % Retrieve the parameter values
        param_value = net.Learnables.Value{i};
        
        % Prune weights below the threshold
        param_value(abs(param_value) < threshold) = 0;

        % Update the parameter in the network
        net.Learnables.Value{i} = param_value;
    end
end