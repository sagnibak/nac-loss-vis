import numpy as np

class MLP(object):
    """A class that defines a multilayer perceptron neural network by initializing
    its weights and implementing forward propagation. Please note that training must
    be done externally by modifying the weights and biases of the MLP object using
    the updating/mutator methods.
    """
    
    def __init__(self, structure: list, activations: list=None, use_bias: bool=False):
        """Initializes the weights of the neural network.
        
        Parameters
        ----------
        structure: List describing the structure of the multilayer perceptron network in
            the following format: [num_inputs, hidden_1, ... , hidden_last, num_outputs].
        activations: List of functions or callable classes to be applied at the end of
            every layer. Must be `None` for no activations at all, or of length 
            (len(structure) - 1), which is equal to the number of layers.
        use_bias: Boolean specifying whether to use bias in the network or not.
        """
        self._W = []
        self._b = []
        self.num_layers = len(structure) - 1
        if len(activations) != self.num_layers:
            raise ValueError('Length of `activations` must be equal to the number ' + 
                             'of layers in the model.')
        self.activations = activations
        
        for i in range(self.num_layers):
            self._W.append(np.random.normal(loc=0.0, scale=1.0,
                                           size=(structure[i], structure[i + 1])))
            if use_bias:
                self._b.append(np.full(fill_value=0., shape=(structure[i + 1])))
                
    def __call__(self, inputs: np.array):
        """Forward propagation.
        
        Parameters
        ----------
        inputs: Numpy ndarray of inputs with shape (batch_size, num_inputs).
        
        Returns
        -------
        Numpy ndarray of results with shape (batch_size, num_inputs).
        """
        x = inputs
        
        for i in range(self.num_layers):
            x = np.dot(x, self._W[i])
            
            if self._b:  # the empty list is False
                x = x + self._b[i]
                
            if self.activations[i]:
                x = self.activations[i](x)
        
        return x
        
    def update_W(self, delta_W, layer: int):
        """Updates `self._W` by adding `delta_W` to it.
        
        Parameters
        ----------
        delta_W: Value to add to `self._W`.
        layer: Index of layer for weight update.
        """
        self._W[layer] += delta_W
        
    def update_b(self, delta_b, layer: int):
        """Updates `self._b` by adding `delta_b` to it.
        
        Parameters
        ----------
        delta_b: Value to add to `self._b`.
        layer: Index of layer for bias update.
        """
        self._b[layer] += delta_b


class NAC(object):
    """A class that defines a neural accumulator, as defined in this paper:
    https://arxiv.org/abs/1808.00508
    Please note that training must be done externally by updating W_hat and
    M_hat using the update methods.
    """
    
    def __init__(self, in_units: int, out_units: int):
        """Initializes the weights of the neural accumulator.
        
        Parameters
        ----------
        in_units: Number of inputs.
        out_units: Number of output units.
        """
        self._W_hat = np.random.normal(loc=0.0, scale=1.0, size=(in_units, out_units))
        self._M_hat = np.random.normal(loc=0.0, scale=1.0, size=(in_units, out_units))
        
    def __call__(self, inputs: np.array):
        """Forward propagation.
        
        Parameters
        ----------
        inputs: Numpy ndarray of inputs with shape (batch_size, in_units)
        
        Returns
        -------
        Numpy ndarray of results with shape (batch_size, out_units).
        """
        self.W = np.multiply(NAC._tanh(self._W_hat), NAC._sigmoid(self._M_hat))
        return np.dot(inputs, self.W)
    
    def update_W_hat(self, delta_W_hat):
        """Adds `delta_W_hat` to `_W_hat`.
        Parameters
        ----------
        delta_W_hat: Value to add to `self._W_hat`.
        """
        self._W_hat += delta_W_hat
        
    def update_M_hat(self, delta_M_hat):
        """Adds `delta_M_hat` to `_M_hat`.
        Parameters
        ----------
        delta_M_hat: Value to add to `self._M_hat`.
        """
        self._M_hat += delta_M_hat
    
    @staticmethod
    def _sigmoid(x):
        """Sigmoid/logistic function.
        
        >>> nac = NAC(1, 1)
        >>> a = np.array([[-100., 0., 100.], [-1.0, 0.3, 1.0]])
        >>> nac._sigmoid(a) < np.array([[0.01, 0.6, 1.01], [0.5, 0.8, 0.5]])
        array([[ True,  True,  True],
               [ True,  True, False]])
        """
        return 1 / (1 + np.exp(-1. * x))
    
    @staticmethod
    def _tanh(x):
        """Tanh function."""
        return np.tanh(x)