"""
This file contains the interface for the neural network;
import the functions/classes implemented here for use in any
other project/code
"""

from nn_imports import *

# so you can call quickWaffle from this file itself
from quick_waffle import quickWaffle


class Waffle:
    def __init__(self):
        """
        Create a new waffle instance. The user doesn't need to provide any data
        when creating the instance; all parameters will be collected through
        the recipe() method\n

        So the table manners for these waffles are:\n
         - create waffle instance  `model = Waffle()`\n
         - give recipe to make the waffle `model.recipe([layers])`\n
         - cook the waffle `model.cook(training_data, hyperparameters)`\n
         - taste the waffle `model.taste(test_data)`\n
         - try again `model.retry()`
        """
        # create an empty list of layers, will be updated by the recipe()
        self.layers = []

        # create an empty dict of params, will be updated by recipe()
        self.parameters = {}

        # create dict of costs, will be updated by cook()
        self.costs = {}
    

    @staticmethod
    def _add_dense_layer(params, prev_params):
        # return a dense layer with given parameters
        if "initialization" in params:

            # if user provided initialization parameters, use it
            if params["initialization"] == "zeros":
                return initialize_zeros([prev_params, params["size"]])
            elif params["initialization"] == "random":
                return initialize_random([prev_params, params["size"]])
            elif params["initialization"] == "xavier":
                return initialize_xavier([prev_params, params["size"]])
            elif params["initialization"] == "he":
                return initialize_he([prev_params, params["size"]])
            
            # wow, you provided invalid initialization values
            else:
            
                # we don't know what you want, we'll give nothing
                return None
        else:

            # user didn't provide any value, default to He initialization
            return initialize_he([prev_params, params["size"]])

    @staticmethod
    def _add_conv_layer(kernel, stride):
        # return a convolutional layer with given parameters
        NotImplemented
    
    @staticmethod
    def _add_pool_layer(params, prev_params):
        # return a pooling layer with given parameters
        NotImplemented

    @staticmethod
    def _add_rnn_layer():
        # return a recurrent layer with given parameters
        NotImplemented

    @staticmethod
    def _add_lstm_layer():
        # return an LSTM layer with given parameters
        NotImplemented
    
    @staticmethod
    def _add_mod_layer(layer):
        # add the given modular layer to the network
        # this function is defined externally by the user, so the user
        # themselves need to take care of the input and output sizes
        NotImplemented
    
    def _kernel_flatten(self):
        # convert the activations from previous (convolutional)
        # layer into 1D array for input to dense layer
        NotImplemented

    def recipe(self, recipe: list):
        """
        Add the given waffle recipe to Waffle properties\n
        The recipe should be provided as:\n
        ``` recipe
        recipe = [
            {
                "type": "input"
                "size": SIZE OF DATASET
            },
            {
                "type": "liege"
                "size": SIZE OF DENSE LAYER,
                "initialization": defaults to He initialization"
            },
            {
                "type": "krumkake",
                "kernel": CONVOLUTION KERNEL SIZE,
                "pad": DATA PADDING,
                "stride": CONVOLUTION STRIDE SIZE
            },
            {
                "type": "cone",
                "mode": MAX/AVG,
            },
            {
                "type": stroopwafel",
                "size": NUMBER OF RECCURENT CELLS
            },
            {
                "type": "honingwafel",
                "size": NUMBER OF LSTM CELLS
            },
            {
                "type": "pizelle",
                "function": MODULAR FUNCTION NAME
            },
            {
                "type": "output",
                "mode": SIGMOID/SOFTMAX,
                "size": SIZE OF OUTPUT LAYER (defaults to 1 if mode is sigmoid)
            }
        ]
        ```\n
        For more details about how to write the waffle recipe, check out the
        source code documentation on the project's GitHub page
        """
        self.layers.append({
            "Layer": "Input",
            "Input Size": recipe[0]["size"]
        })
        for num, ingredient in enumerate(recipe[1:]):  # assuming the first entry is input layer
            # to make the code cleaner
            layer = ingredient["type"]
            prev_params = recipe[num-1]["size"]
            
            # add dense layer
            if layer == "liege":
                params = self._add_dense_layer(ingredient, prev_params)
                self.layers.append({
                    "Layer": num + 1,
                    "Type": "Liege",
                    "parameters": params})
                self.parameters[f"layer {num+1}"] = params
            
            # add convolutional layer
            elif layer == "krumkake":
                self.layers.append(self._add_conv_layer(params[0], params[1]))
            
            # add pooling layer
            elif layer == "cone":
                self.layers.append(self._add_pool_layer(ingredient, prev_params))
            
            # add simple recurrent layer
            elif layer == "stroopwafel":
                self.layers.add(self._add_rnn_layer(params))
            
            # add LSTM recurrent layer
            elif layer == "honingwafel":
                self.layers.append(self._add_lstm_layer(params))
            
            # add a modular layer
            elif layer == "pizelle":
                self.layers.append(self._add_mod_layer(params))
            
            # add the output layer
            elif layer == "output":
                # you want a single output, we'll add a sigmoid layer
                if ingredient["mode"] == "sigmoid":
                    self.layers.append({
                        "Layer": "Output",
                        "Mode": "Sigmoid",
                        "Output Values": 1})
                
                # just making sure you didn't give invalid values
                elif ingredient["mode"] == "softmax" and ingredient["size"] > 1:
                    self.layers.append({
                        "Layer": "Output",
                        "Mode": "Softmax",
                        "Output Values": ingredient["size"]})
                
                # so you did give invalid output size
                else:
                    print("Invalid Output Layer Size, please enter recipe again")
                    
                    # we don't want to keep appending layers
                    self.layers = []
                    break
            
            # don't know this type of waffle :/
            else:
                print("Invalid ingredient name, please enter recipe again")
                
                # we'll pretend we didn't see anything
                self.layers = []
                break
            
            if params is None:

                # if, for any reason, we were unable to make the layer,
                # we have printed out the reason already, no we'll make sure
                # there's no clutter
                self.layers = []
                break
    
    def cook(self, train_features: np.ndarray, train_labels: np.ndarray,
          epochs: int = 10000, batch_size: int = 64, real_time_report: bool = False,
          initialization: str = "he", optimization: str = "adam",
          regularization: str = "none", alpha: float = 0.5, beta: float = 0.9,
          gamma: float = 0.999, delta: float = 0.0, epsilon: int = 1e-8,
          lamda: float = 0.5, kappa: float = 0.5):
        """
        Bake the given waffle for given amount of time\n
        Also, if you're too hungry, provide real-time cooking progress\n
        The parameters available for tuning are:\n
        :param `train_features`: training feature set (numpy ndarray)
        :param `train_labels`: training labels (numpy array)
        :param `epochs`: iterations of model training (int)
        :param `batch_size`: size of mini-batch (int)
        :param `real_time_report`: set to True to see training progress (bool)
        :param `initialization`: zero/random/he (str)
        :param `optimization`: gd/momentum/rmsprop/adam (str)
        :param `regularization`: none/L2 (str)
        :param `alpha`: gradient descent learning rate (float)
        :param `beta`: momentum parameter (float)
        :param `gamma`: RMSProp parameter (float)
        :param `delta`: learning rate decay parameter (float)
        :param `epsilon`: Adam parameter (int)
        :param `lamda`: L2 regularization parameter (float)
        :param `kappa`: dropout probability parameter (float)\n
        For more information about what each parameter means, check out
        the source code documentation on the project's GitHub page
        """
        # make sure we have model structure data
        if self.layers == []:
            return "No recipe provided, please add a recipe for the waffle using recipe() method"
        
        # finished training, update values in layers for visualization
        for i, layer in enumerate(self.layers[1:]):
            layer["parameters"] = self.parameters[f"layer {i+1}"]
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
        and cook with the same recipe again\n

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
    
    def __repr__(self):
        # Get a glimpse of your waffle
        # DrawNN([20] + self.layers[1:]).draw()
        layers = []
        for layer in self.layers:
            layers.append(str(layer))
        return "\n".join(layers)


def main():
    model = Waffle()
    recipe = [
        {
            "type": "input",
            "size": 20
        },
        {
            "type": "liege",
            "size": 5,
            "initialization": "he"
        },
        {
            "type": "liege",
            "size": 7,
            "initialization": "random"
        },
        {
            "type": "output",
            "mode": "sigmoid",
            "size": 1
        }
    ]
    model.recipe(recipe)
    model.cook()
    print(model)

# main()
