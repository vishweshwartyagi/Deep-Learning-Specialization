import numpy as np
import matplotlib.pyplot as plt
import h5py


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    a -- sigmoid(z)
    cache -- returns z as well, useful during backpropagation
    """
    
    a = 1 / (1 + np.exp(-z))
    cache = z
    
    return a, cache


def relu(z):
    """
    Implement the RELU function.

    Arguments:
    z -- Output of the linear layer, of any shape

    Returns:
    a -- Post-activation parameter, of the same shape as z
    cache -- a python dictionary containing "a" ; stored for computing the backward pass efficiently
    """
    
    a = np.maximum(0, z)
    
    assert(a.shape == z.shape)
    
    cache = z
    return a, cache


def relu_backward(da, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    da -- post-activation gradient, of any shape
    cache -- 'z' where we store for computing backward propagation efficiently

    Returns:
    dz -- Gradient of the cost with respect to z
    """
    
    z = cache
    dz = np.array(da, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dz[z <= 0] = 0
    
    assert (dz.shape == z.shape)
    
    return dz

def sigmoid_backward(da, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    da -- post-activation gradient, of any shape
    cache -- 'z' where we store for computing backward propagation efficiently

    Returns:
    dz -- Gradient of the cost with respect to z
    """
    
    z = cache
    
    s = 1/(1+np.exp(-z))
    dz = da * s * (1-s)
    assert (dz.shape == z.shape)
    
    return dz


def load_dataset():
    with h5py.File('data/train_catvnoncat.h5', "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    with h5py.File('data/test_catvnoncat.h5', "r") as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


def initialize_parameters(n_0, n_1, n_2):
    """
    Argument:
    n_0 -- size of the input layer
    n_1 -- size of the hidden layer
    n_2 -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    w1 -- weight matrix of shape (n_0, n_1)
                    b1 -- bias vector of shape (n_1, 1)
                    w2 -- weight matrix of shape (n_1, n_2)
                    b2 -- bias vector of shape (n_2, 1)
    """
    
    np.random.seed(1) # set up a seed so that output matches everytime
    
    w1 = np.random.randn(n_1, n_0).T * 0.01
    b1 = np.zeros(shape=(n_1, 1))
    w2 = np.random.randn(n_2, n_1).T * 0.01
    b2 = np.zeros(shape=(n_2, 1))
    
    assert (w1.shape == (n_0, n_1))
    assert (b1.shape == (n_1, 1))
    assert (w2.shape == (n_1, n_2))
    assert (b2.shape == (n_2, 1))
    
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}
    
    return parameters


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "w1", "b1", ..., "wL", "bL":
                    wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network
    
    for l in range(1, L): 
        parameters['w' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]).T / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros(shape=(layer_dims[l], 1))
        
        assert(parameters['w' + str(l)].shape == (layer_dims[l-1], layer_dims[l]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


def linear_forward(a , w, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    a -- activations from previous layer (or input data): (size of previous layer, number of examples)
    w -- weights matrix: numpy array of shape (size of previous layer, size of current layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "a", "w" and "b" ; stored for computing the backward pass efficiently
    """
    
    z = np.dot(w.T, a) + b
    
    cache = (a, w, b)
    
    return z, cache


def linear_activation_forward(a_prev, w, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    a_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    w -- weights matrix: numpy array of shape ( size of previous layer, size of current layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    a -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = sigmoid(z)
    
    elif activation == "relu":
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = relu(z)

    
    assert (a.shape == (w.shape[1], a_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return a, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    aL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    a = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        a_prev = a 
        a, cache = linear_activation_forward(a_prev, parameters["w" + str(l)], parameters["b" + str(l)], activation = "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    aL, cache = linear_activation_forward(a, parameters["w" + str(L)], parameters["b" + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    assert(aL.shape == (1, X.shape[1]))
            
    return aL, caches


def compute_cost(aL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    aL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]
    if Y.shape[1] > 1:
        Y = Y.T 

    # Compute loss from aL and y
    cost = -(1./m) * ( np.dot(Y.T, np.log(aL.T)) + np.dot((1-Y).T, np.log(1-aL.T)) )[0, 0]
    
    assert(cost.shape == ())
    
    return cost


def linear_backward(dz, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dz -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (a_prev, w, b) coming from the forward propagation in the current layer

    Returns:
    da_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dw -- Gradient of the cost with respect to W (current layer l), same shape as w
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    
    a_prev, w, b = cache
    m = a_prev.shape[1]

    dw = (1/m) * np.dot(a_prev, dz.T)
    db = (1/m) * np.sum(dz, axis=1, keepdims=True)
    da_prev = np.dot(w, dz)
    
    assert (da_prev.shape == a_prev.shape)
    assert (dw.shape == w.shape)
    assert (db.shape == b.shape)
    
    return da_prev, dw, db


def linear_activation_backward(da, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    da -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    da_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as a_prev
    dw -- Gradient of the cost with respect to w (current layer l), same shape as w
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dz = relu_backward(da, activation_cache)
        da_prev, dw, db = linear_backward(dz, linear_cache)
        
    elif activation == "sigmoid":
        dz = sigmoid_backward(da, activation_cache)
        da_prev, dw, db = linear_backward(dz, linear_cache)
    
    return da_prev, dw, db


def L_model_backward(aL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    aL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["da" + str(l)] = ... 
             grads["dw" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = aL.shape[1]
    if Y.shape[1] > 1:
        Y = Y.T
   
    # Initializing the backpropagation
    daL = - ( Y/aL.T - (1-Y)/(1-aL.T) ).T
    
    # Lth layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[L-1]
    grads["da" + str(L)], grads["dw" + str(L)], grads["db" + str(L)] = linear_activation_backward(daL, current_cache, "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        da_prev_temp, dw_temp, db_temp = linear_activation_backward(grads["da" + str(l + 2)], current_cache, "relu")
        grads["da" + str(l + 1)] = da_prev_temp
        grads["dw" + str(l + 1)] = dw_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["w" + str(l+1)] = parameters["w" + str(l+1)] - learning_rate * grads["dw" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l + 1)]
        
    return parameters


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p


def print_mislabeled_images(classes, X, Y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    Y -- true labels
    p -- predictions
    """
    a = p + Y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[Y[0,index]].decode("utf-8"))