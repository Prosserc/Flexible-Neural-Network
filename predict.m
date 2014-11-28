function p = predict(Theta1, Theta2, Theta3, X)
   %PREDICT Predict the label of an input given a trained neural network
   %   p = PREDICT(Theta1, Theta2, Theta3, X) outputs the predicted label 
   %   of X given the trained weights of a neural network

   m = size(X, 1);
   p = zeros(m, 1);

   h1 = sigmoid([ones(m, 1) X] * Theta1');
   h2 = sigmoid([ones(m, 1) h1] * Theta2');
   h3 = sigmoid([ones(m, 1) h2] * Theta3');

   % if only one output unit...
   if size(h3, 2) == 1,
      % predict 1 if final layer output > 0.5
      p = [h3 > 0.5];
   else
      % get index of max value corresponding to class
      [dummy, p] = max(h3, [], 2);
   end
   
end
