# shiny-waffle
Simple neural network written in Python while studying the Deep Learning Specialization at Coursera. This (logistic regression) network also contains implementation of He initialization, L2 regularization, dropout regularization (which I haven't actually tested and am pretty sure doesn't work), along with Gradient descent with momentum, RMSProp and Adam optimization methods. Pay attention to the parameter notations used in the code (I prefer calling parameters &beta; and &gamma; instead of &beta;<sub>1</sub> and &beta;<sub>2</sub>).

## Parameters:

layers: a list of number of hidden units in the hidden layers you want to implement. Say you want a network with 3 hidden layers, first layer containing 4 hidden units, second having 3 units and third having 5 units, then the `layers` parameter will be `[4, 3, 5]`. Keep in mind that you only need to provide the dimensions for the hidden layers, the code automatically adds the input and output layers (the output layer is default to `1`, set `softmax=True` so that it gets the shape of number of unique elements in `y_train`.<br>
&alpha;: gradient descent learning rate (generally set between 0.1 and 0.001)<br>
&beta;: momentum parameter for gradient descent with momentum and Adam optimization (generally set to 0.9)<br>
&gamma;: RMS parameter for RMSProp and Adam optimization (generally set to 0.999)<br>
&delta;: learning rate decay parameter (generally set to 0.95). This algorithm works with the equation (&alpha; = &delta;<sup>epoch number</sup> x &alpha;<sub>0</sub>), so &delta; behaves chaotically between 0.9 and 1.<br>
&epsilon;: Adam optimization parameter to prevent division by 0 (set to 10<sup>-8</sup>, you don't generally need to change this)<br>
&lambda;: L2 regularization parameter (value needs to be set experimentally for each use case)<br>
