import matplotlib.pyplot as plt
import numpy as np 


def sigmoid(Z):
    return 1. / (1. + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def plot_cost_over_iterations(n_iterations, train_costs, test_costs):
    """
    Plots the training and test costs over iterations on the same plot.

    Arguments:
    n_iterations -- Number of iterations (epochs)
    train_costs -- List of training costs over iterations
    test_costs -- List of test costs over iterations
    """
    
    plt.figure(figsize=(6, 4))  # Set the figure size

    # Plot the training costs in blue
    plt.plot(range(n_iterations), train_costs, color='b', linewidth=2, label='Training Cost')

    # Plot the test costs in red
    plt.plot(range(n_iterations), test_costs, color='r', linewidth=2, label='Test Cost')

    # Customizing the plot
    plt.xlabel('Iterations', fontsize=10)
    plt.ylabel('Cost', fontsize=10)
    plt.title(f'Cost Function Over Iterations', fontsize=12)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid for better readability
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add a legend to differentiate between Training and Test cost
    plt.legend(loc='best', fontsize=10)

    # Display the plot
    plt.tight_layout()
    plt.show()
    
    
    
def predict(X, Y, opt_params):
    """
    Predicts the labels for the input data X using the optimized parameters.

    Arguments:
    X -- Input data of shape (input_size, number of examples)
    Y -- True labels of shape (1, number of examples)
    opt_params -- Dictionary containing the optimized parameters "W1", "b1", ..., "WL", "bL"

    Returns:
    predictions -- Predictions for each example (0 or 1)
    accuracy -- Accuracy of the model's predictions, compared to the true labels
    """

    m = X.shape[1]  # number of examples
    L = len(opt_params) // 2  # number of layers in the neural network
    A = X
    
    # Forward propagation through all layers
    for l in range(1, L):
        Z = np.dot(opt_params['W' + str(l)], A) + opt_params['b' + str(l)]
        A = relu(Z)  # Use ReLU activation for hidden layers
    
    # Output layer (using sigmoid activation)
    ZL = np.dot(opt_params['W' + str(L)], A) + opt_params['b' + str(L)]
    AL = sigmoid(ZL)
    
    # Convert probabilities to binary predictions
    predictions = (AL > 0.5).astype(int)
    
    # Calculate the accuracy
    accuracy = np.mean(predictions == Y) * 100
    
    return predictions, accuracy