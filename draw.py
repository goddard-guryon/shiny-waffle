"""
Code author: Oliver Wilken (https://stackoverflow.com/a/37366154)
Added features: 
    - Display annotation on hovering on a neuron
    - Display convolutional, recurrent, pooling & modular layers
      in a different style
"""

from matplotlib import pyplot as plt
from math import sin, cos, atan


class Neuron:
    def __init__(self, x, y):
        # set x,y-coordinates of the circle
        self.x = x
        self.y = y
    
    def draw(self, radius):
        # create the circle
        circle = plt.Circle((self.x, self.y), radius=radius, fill=False)
        plt.gca().add_patch(circle)


class Layer:
    def __init__(self, network, num_neurons, num_neurons_max):
        # set parameters: distance between layers
        self.dist_vert_layers = 6
        # distance between neurons
        self.dist_hori_neuron = 2
        # size of neurons
        self.neuron_radius = 0.5
        # number of neurons in the widest layer
        self.num_neurons_max = num_neurons_max
        # properties of previous layer
        self.prev_layer = self.__get_prev_layer(network)
        # y-coordinates for the current layer
        self.y = self.__calc_y_pos()
        # finally, create the neurons in the layer
        self.neurons = self.__init_neurons(num_neurons)
    
    def __init_neurons(self, num_neurons):
        neurons = []
        # get the x-coordinates for this layer
        x = self.__get_centered_x(num_neurons)
        for _ in range(num_neurons):
            # create neurons
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.dist_hori_neuron
        return neurons
    
    def __get_centered_x(self, num_neurons):
        # get x-coordinates of this layer at y=0
        return self.dist_hori_neuron * (self.num_neurons_max - num_neurons) / 2
    
    def __calc_y_pos(self):
        # get y-coordinates for this layer
        if self.prev_layer:
            # if we're in hidden/output layer, change the coordinates
            return self.prev_layer.y + self.dist_vert_layers
        else:
            # we're at input layer, start at the origin
            return 0
    
    def __get_prev_layer(self, network):
        if len(network.layers) > 0:
            # we've already worked on some layers, return the latest one
            return network.layers[-1]
        else:
            # we haven't done anything yet
            return None
    
    def __line_bw_neurons(self, n_1, n_2):
        # create a line between 2 neurons by calculating x,y-coordinates
        # get angle between the 2 neurons
        angle = atan((n_2.x - n_1.x) / float(n_2.y - n_1.y))
        # find change in x & y axes base on the angle
        x_adj = self.neuron_radius * sin(angle)
        y_adj = self.neuron_radius * cos(angle)
        # create the line
        line = plt.Line2D((n_1.x - x_adj, n_2.x + x_adj), (n_1.y - y_adj, n_2.y + y_adj))
        # default line width is too much
        line.set_linewidth(0.7)
        plt.gca().add_line(line)
    
    def draw(self, layer_type = 0):
        # draw the whole layer
        for neuron in self.neurons:
            # draw every neuron in this layer
            neuron.draw(self.neuron_radius)
            if self.prev_layer:
                # if there is a previous layer, make connections with it
                for prev_neuron in self.prev_layer.neurons:
                    self.__line_bw_neurons(neuron, prev_neuron)
        
        # create layer annotations
        x_text = self.num_neurons_max * self.dist_hori_neuron
        if layer_type == 0:
            plt.text(x_text, self.y, "Input Layer", fontsize=12)
        elif layer_type == -1:
            plt.text(x_text, self.y, "Output Layer", fontsize=12)
        else:
            plt.text(x_text, self.y, f"Hidden Layer {layer_type}", fontsize=12)


class Network:
    def __init__(self, num_neurons_max):
        # initialize parameters
        self.num_neurons_max = num_neurons_max
        self.layers = []
        self.layer_type = 0
    
    def add_layer(self, num_neurons):
        # add a neuron layer to the network
        layer = Layer(self, num_neurons, self.num_neurons_max)
        self.layers.append(layer)
    
    def draw(self):
        # draw the network
        plt.figure()
        for i, layer in enumerate(self.layers):
            # if we're at last layer, change the value for annotation
            if i == len(self.layers) - 1:
                i = -1
            layer.draw(i)
        plt.axis("Scaled")
        plt.axis("Off")
        plt.title(f"Your {len(self.layers)}-layer NN waffle (^ᴗ^)", fontsize=15)
        plt.show()


class DrawNN:
    def __init__(self, network):
        # get the neural network
        self.network = network
    def draw(self):
        # draw the neural network
        widest_layer = max(self.network)
        network = Network(widest_layer)
        for l in self.network:
            network.add_layer(l)
        network.draw()


def annotate():
    import matplotlib.pyplot as plt
    import numpy as np; np.random.seed(1)

    x = np.sort(np.random.rand(15))
    y = np.sort(np.random.rand(15))
    names = np.array(list("ABCDEFGHIJKLMNO"))
    norm = plt.Normalize(1,4)
    cmap = plt.cm.RdYlGn

    fig,ax = plt.subplots()
    line, = plt.plot(x,y, marker="o")

    annot = ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        x,y = line.get_data()
        annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
        text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                            " ".join([names[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = line.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()
