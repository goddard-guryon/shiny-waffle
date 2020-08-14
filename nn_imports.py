def import_all():
    """
    Imports all the necessary packages
    I know this is hideous, but having a dozen lines of code, that make
    no sense, right at the top of the file, looks even more hideous to
    me. So, please forgive me for this one :)
    """
    # from external resources, we'll need numpy and matplotlib only
    global np, plt, col
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as col

    # this also helps as a directory of functions
    names = {
        "activations": [
            "sigmoid",
            "softmax"
        ],
        "initialize": [
            "initialize_zeros",
            "initialize_random",
            "initialize_xavier",
            "initialize_he",
            "mini_batches",
            "initialize_velocity",
            "initialize_rms",
            "initialize_adam"
        ],
        "forward_prop": [
            "linear_forward"
        ],
        "backprop": [
            "linear_backward",
            "linear_backward_with_L2"
        ],
        "cost": [
            "cross_entropy_cost_mini",
            "cost_with_L2"
        ],
        "update": [
            "update_parameters",
            "update_parameters_with_momentum",
            "update_parameters_with_rms",
            "update_parameters_with_adam"
        ],
        "draw": [
            "DrawNN"
        ]
    }

    # import everything given in the dictionary
    for library, imports in names.items():
        for name in imports:
            try:
                # I looked for a solution to this for like 6 hours, lol
                exec(f"global {name}; from {library} import {name}")
            
            except ImportError:
                # in case you rename a function and forget to change it here
                print(f"Missing function {library}.{name}, please check the source files")
                return False
    
    # the return statement doesn't really have a use, but looks cool
    return True


import_all()
