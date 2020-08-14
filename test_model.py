import numpy as np
import sklearn.datasets as data
import matplotlib.pyplot as plt
from forward_prop import linear_forward
from Waffle import quickWaffle
from predict import taste_waffle

def load_dataset():
    np.random.seed(3)
    train_X, train_Y = data.make_moons(n_samples=300, noise=.2) #300 #0.2 
    # Visualize the data
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    
    return train_X, train_Y


def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Predict using forward propagation and a classification threshold of 0.5
    a3, _ = linear_forward(X, parameters)
    predictions = (a3 > 0.5)
    return predictions


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap="viridis")
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y[0], cmap="rainbow_r", alpha=0.5)
    plt.show()


X_train, y_train = load_dataset()
layers = [6, 7, 8, 7, 6]

model, costs = quickWaffle(layers, X_train, y_train, num_epochs=4001, batch_size=128,
                                alpha=0.0075, beta=0.85, gamma=0.99999, delta=0.9995, lamda=0.15,
                                print_cost=True, regularization="L2")
preds = taste_waffle(X_train, y_train, model)

plt.title("Moons Waffle with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(model, x.T), X_train, y_train)
