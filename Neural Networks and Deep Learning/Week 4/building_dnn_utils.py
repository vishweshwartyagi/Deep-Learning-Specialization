import numpy as np

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


