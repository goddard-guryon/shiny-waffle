# code taken from https://gist.github.com/anbrjohn/7116fa0b59248375cd0c0371d6107a59

"""
Created by @author: craffel
Modified on Sun Jan 15, 2017 by anbrjohn
Modifications: 
    -Changed xrange to range for python 3
    -Added functionality to annotate nodes
Modified on Wed Jul 15, 2020 by goddard-guryon
Modifications:
    -Added code to create annotations for arbitrarily sized neural networks
    -Decreased line width of edges
"""

import matplotlib.pyplot as plt

def draw_neural_net(ax, left, right, bottom, top, layers):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2], ['x1', 'x2','x3','x4'])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
        - layer_text : list of str
            List of node annotations in top-down left-right order
    '''

    # create list of annotations for arbitrarily sized neural networks
    layer_sizes = [21] + layers[1:]
    layer_text = []
    for i, node in enumerate(layer_sizes):
        for j in reversed(range(node)):
            if i == 0:
                if j == 20:
                    layer_text.append("...")
                else:
                    layer_text.append(f"$X_{{{j+1}}}$")
            elif node == layer_sizes[-1]:
                layer_text.append("$\hat{y}$")
            else:
                layer_text.append(f"$h_{{{str(i+1) + ',' + str(j+1)}}}$")
    
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    ax.axis('off')
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            x = n*h_spacing + left
            y = layer_top - m*v_spacing
            circle = plt.Circle((x,y), v_spacing/2.5,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            # Node annotations
            if layer_text:
                text = layer_text.pop(0)
                plt.annotate(text, xy=(x, y), zorder=5, ha='center', va='center')


    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k', linewidth=0.75)
                ax.add_artist(line)
