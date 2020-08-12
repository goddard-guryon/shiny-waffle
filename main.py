import numpy as np
from typing import Union, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as col
from activations import sigmoid, softmax
from initialize import mini_batches, initialize_adam, initialize_he, initialize_random, initialize_rms, initialize_velocity, initialize_zeros
from linear import lin_forward_prop, lin_backward_prop, lin_backward_prop_with_L2
from cost import cross_entropy_cost_mini, cost_with_L2
from update import update_parameters, update_parameters_with_momentum, update_parameters_with_rms, update_parameters_with_adam
from draw import draw_neural_net


def make_waffle(layers: list, X_train: np.ndarray, y_train: np.ndarray,
          num_epochs: int = 10000, batch_size: int = 64, print_cost: bool = False,
          initialization: str = "he", optimization: str = "adam",
          regularization: str = "none", alpha: float = 0.5, beta: float = 0.9,
          gamma: float = 0.999, delta: float = 0.0, epsilon: int = 1e-8,
          lamda: float = 0.5, kappa: float = 0.5, softmax: bool = False):
    """
    Initialize neural network model and fit to given data
    :param layers: dimensions of neural network eg- [3, 5, 2] (list)
    ONLY ADD THE SIZES OF HIDDEN LAYERS, the first and last layer is added automatically
    :param X_train: training feature set (numpy ndarray)
    :param y_train: training labels (numpy array)
    :param num_epochs: iterations of model training (int)
    :param batch_size: size of mini-batch (int)
    :param print_cost: set to True to see training progress (bool)
    :param initialization: zero/random/he (str)
    :param optimization: gd/momentum/rmsprop/adam (str)
    :param regularization: none/L2 (str)
    :param alpha: gradient descent learning rate (float)
    :param beta: momentum parameter (float)
    :param gamma: RMSProp parameter (float)
    :param delta: learning rate decay parameter (float)
    :param epsilon: Adam parameter (int)
    :param lamda: L2 regularization parameter (float)
    :param kappa: dropout probability parameter (float)
    :param softmax: whether the model requires multi-class regression (bool)
    """
    # initialize parameters
    costs = []
    t = 0
    layers = [X_train.shape[0]] + layers + [1]
    m = X_train.shape[1]

    if initialization == "random":
        parameters = initialize_random(layers)
    elif initialization == "he":
        parameters = initialize_he(layers)
    else:
        parameters = initialize_zeros(layers)


    if optimization == "momentum":
        velocity = initialize_velocity(parameters)
    elif optimization == "rmsprop":
        rms = initialize_rms(parameters)
    elif optimization == "adam":
        velocity, rms = initialize_adam(parameters)
    else:
        pass

    print(f"Preparing to cook a waffle for {num_epochs} minutes...")


    # start iteration
    for epoch in range(num_epochs):

        # create the mini-batches
        batches = mini_batches(X_train, y_train, batch_size)
        total_cost = 0

        # implement learning rate decay if delta is specified by the user
        if delta > 0.0:
            alpha_0 = alpha * delta**epoch

        for batch in batches:
            X, y = batch

            # forward propagation
            AL, cache = lin_forward_prop(X, parameters, softmax)

            # compute cost value
            if regularization == "L2":
                total_cost += cost_with_L2(AL, parameters, y, lamda)
            else:
                total_cost += cross_entropy_cost_mini(AL, y_train)

            # backward propagation
            if regularization == "L2":
                gradients = lin_backward_prop_with_L2(AL, y, cache, lamda, parameters, softmax)
            else:
                gradients = lin_backward_prop(AL, y, cache, softmax)

            # update parameter values
            if optimization == "momentum":
                parameters, velocity = update_parameters_with_momentum(parameters, gradients,
                                                             velocity, alpha_0, beta)
            elif optimization == "rmsprop":
                parameters, rms = update_parameters_with_rms(parameters, gradients,
                                                        rms, alpha_0, gamma)
            elif optimization == "adam":
                t += 1
                parameters, velocity, rms = update_parameters_with_adam(parameters, gradients,
                                                         velocity, rms, alpha_0,
                                                         beta, gamma, epsilon, t)
            else:
                parameters = update_parameters(parameters, gradients, alpha_0)

        # check average cost
        avg_cost = total_cost / m

        # print out some updates
        if epoch % 1000 == 0:
            costs.append(avg_cost)
            if print_cost:
                print(f"Waffle texture after {epoch} minutes: {avg_cost}")

    # let's show how the model looks
    if print_cost:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        draw_neural_net(ax, 0.05, 0.95, 1, 0, layers)
        plt.title(f"Your {len(layers)}-layer NN waffle (^ᴗ^)")
        plt.show()

    # return the important stuff
    return parameters, costs



class Waffle():
    def __init__(self):
        """
        Create a new waffle instance. The user doesn't need to provide any data
        when creating the instance; all parameters will be collected through
        the recipe() method

        So the table manners for these waffles are:
         - create waffle instance  model = Waffle()
         - give recipe to make the waffle model.recipe([layers])
         - cook the waffle model.cook(parameters)
         - taste the waffle model.taste(test_data)
         - try again model.retry()
        """
        # create an empty list of layers, will be updated by the recipe()
        self.layers = []

        # create an empty dict of params, will be updated by recipe()
        self.parameters = {}

        # create dict of costs, will be updated by cook()
        self.costs = {}
    

    @staticmethod
    def add_dense_layer(size):
        # return a dense layer with given parameters
        NotImplemented

    @staticmethod
    def add_conv_layer(kernel, stride):
        # return a convolutional layer with given parameters
        NotImplemented

    @staticmethod
    def add_rnn_layer():
        # return a recurrent layer with given parameters
        NotImplemented

    @staticmethod
    def add_lstm_layer():
        # return an LSTM layer with given parameters
        NotImplemented
    
    @staticmethod
    def add_mod_layer(layer):
        # add the given modular layer to the network
        # this function is defined externally by the user, so the user
        # themselves need to take care of the input and output sizes
        NotImplemented

    def recipe(self, recipe: dict):
        """
        Add the given waffle recipe to Waffle properties
        The recipe should be provided as:
        recipe = [
            {"input": SIZE OF DATASET},
            {"liege": SIZE OF DENSE LAYER},
            {"krumkake": [CONV LAYER KERNEL SIZE, STRIDE SIZE]},
            {"stroopwafel": [RECC LAYER PARAMETERS]},
            {"honingwafel": [LSTM LAYER PARAMETERS]},
            {"pizelle": FUNCTION FOR MODULAR LAYER},
            {"output": SIZE OF OUTPUT LAYER}
        ]
        """
        for num, ingredient in enumerate(recipe[1:]):  # assuming the first entry is input layer
            # to make the code cleaner
            name, params = ingredient.items()
            
            # add dense layer
            if name == "liege":
                self.layers.append(self.add_dense_layer(params))
            
            # add convolutional layer
            elif name == "krumkake":
                self.layers.append(self.add_conv_layer(params[0], params[1]))
            
            # add simple recurrent layer
            elif name == "stroopwafel":
                self.layers.add(self.add_rnn_layer(params))
            
            # add LSTM recurrent layer
            elif name == "honingwafel":
                self.layers.append(self.add_lstm_layer(params))
            
            # add a modular layer
            elif name == "pizelle":
                self.layers.append(self.add_mod_layer(params))
            
            # add the output layer
            elif name == "output":
                # you want a single output, we'll add a sigmoid layer
                if params == 1:
                    self.layers.append(sigmoid())
                
                # just making sure you didn't give invalid values
                elif params > 1:
                    self.layers.append(softmax())
                
                # so you did give invalid output size
                else:
                    print("Invalid Output Layer Size, please enter recipe again")
                    
                    # we don't want to keep appending layers
                    self.layers = []
            
            # don't know this type of waffle :/
            else:
                print("Invalid ingredient name, please enter recipe again")
                
                # we'll pretend we didn't see anything
                self.layers = []
    
    def cook(self, train_features: np.ndarray, train_labels: np.ndarray,
          epochs: int = 10000, batch_size: int = 64, real_time_report: bool = False,
          initialization: str = "he", optimization: str = "adam",
          regularization: str = "none", alpha: float = 0.5, beta: float = 0.9,
          gamma: float = 0.999, delta: float = 0.0, epsilon: int = 1e-8,
          lamda: float = 0.5, kappa: float = 0.5):
        """
        Bake the given waffle for given amount of time
        Also, if you're too hungry, provide real-time cooking progress

        :param train_features: training feature set (numpy ndarray)
        :param train_labels: training labels (numpy array)
        :param epochs: iterations of model training (int)
        :param batch_size: size of mini-batch (int)
        :param real_time_report: set to True to see training progress (bool)
        :param initialization: zero/random/he (str)
        :param optimization: gd/momentum/rmsprop/adam (str)
        :param regularization: none/L2 (str)
        :param alpha: gradient descent learning rate (float)
        :param beta: momentum parameter (float)
        :param gamma: RMSProp parameter (float)
        :param delta: learning rate decay parameter (float)
        :param epsilon: Adam parameter (int)
        :param lamda: L2 regularization parameter (float)
        :param kappa: dropout probability parameter (float)
        """
        # make sure we have model structure data
        if self.layers == []:
            return "No recipe provided, please add a recipe for the waffle using recipe() method"
        pass
    
    def taste(self, test_features, test_labels):
        """
        Get your waffle into a taste test using the given
        test data set
        """
        NotImplemented
    
    def retry(self):
        """
        In case you aren't satisfied with the waffle and want to make
        another one, use this method to clean up your waffle parameters
        and cook with the same recipe again

        Again, this method cleans all waffle parameters and costs, but
        keeps the layers data intact. If you also want to change the
        layers data along with parameters and costs, call this method,
        then the dump_recipe() method and fill in recipe() again
        """
        self.parameters = {}
        self.costs = {}
    
    def dump_recipe(self):
        """
        Remove the waffle recipe while keeping parameters and costs intact.
        Once you call this method, Waffle.cook() will not work unless you
        provide a new recipe through the Waffle.recipe() method
        """
        self.layers = []
    
    def __print__(self):
        # Get a glimpse of your waffle
        NotImplemented
