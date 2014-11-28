function g = sigmoidGradient(z)
   %SIGMOIDGRADIENT returns the gradient of the sigmoid function
   %evaluated at z

   % evaluate sigmoid of input once
   g = zeros(size(z));
   
   % use to calc sigmoid gradient for each input element
   tmp_g = sigmoid(z);
   g = tmp_g.*(1-tmp_g);
   
end
