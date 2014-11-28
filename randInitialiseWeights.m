function W = randInitialiseWeights(L_in, L_out)
   %Randomly initialize the weights of a layer with L_in
   %incoming connections and L_out outgoing connections
   % Note: The first row of W corresponds to the parameters for the bias units
   epsilon_init = sqrt(6)/sqrt(L_in+L_out);
   W = rand(L_out, 1 + L_in)*2*epsilon_init-epsilon_init;
end
