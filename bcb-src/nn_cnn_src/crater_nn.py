"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes, printoutput=True):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        
        # include initialization enhancement from network2
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
        # some extra attributes that customize our network
        self.threshold = 0.5 # activation threshold can be changed
        self.epoch = 0       # keeps track of the total epochs of training
        self.history = []    # stores test evaluation after each epoch
        self.validation_history = []
        self.printoutput = printoutput 
       
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def feedforward_flat(self, a):
        """Return the output of the network if ``a`` is input."""
        a = a.reshape((len(a), 1))
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, validation_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        if validation_data: n_validation = len(validation_data)
        n = len(training_data)
        for j in range(epochs):
            self.epoch += 1
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                rs = self.evaluate(test_data)
                tp, fp, fn = rs[0], rs[1], rs[2]
                det, fal, qua = rs[3], rs[4], rs[5]
                acc = rs[6]
                epochdata = (self.epoch, tp, fp, fn, det, fal, qua, acc)
                self.history.append(epochdata)
                if self.printoutput:
                    print (("Test at epoch % 5d: % 7d % 7d % 7d "
                            "% 5.4f % 5.4f % 5.4f % 14.4f") % epochdata)
            if validation_data:
                rs = self.evaluate(validation_data)
                tp, fp, fn = rs[0], rs[1], rs[2]
                det, fal, qua = rs[3], rs[4], rs[5]
                acc = rs[6]
                epochdata = (self.epoch, tp, fp, fn, det, fal, qua, acc)
                self.validation_history.append(epochdata)
                if self.printoutput:
                    print (("Vali at epoch % 5d: % 7d % 7d % 7d "
                            "% 5.4f % 5.4f % 5.4f % 14.4f") % epochdata)
            elif self.printoutput:
                print ("Epoch {0} complete".format(self.epoch))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        th = self.threshold
        test_results = [(int(self.feedforward(x)[0][0] > th), y)
                        for (x, y) in test_data]
        
        # the confusion matrix
        #       TN  FN
        #       FP  TP
        cm = [[0, 0],[0, 0]] 
        for x, y in test_results:
            cm[x][y] += 1
        
        tn, fn = cm[0][0], cm[0][1]
        fp, tp = cm[1][0], cm[1][1]

        try:
            det_r = float(tp) / ( tp + fn )
        except ZeroDivisionError:
            det_r = 0.0
        try:
            fal_r = float(fp) / ( tp + fp )
        except ZeroDivisionError:
            fal_r = 1.0
        try:
            qua_r = float(tp) / ( tp + fp + fn )
        except ZeroDivisionError:
            qua_r = 0.0
        
        acc = float(tp + tn) / len(test_results)
        
        return tp, fp, fn, det_r, fal_r, qua_r, acc

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives partial C_x /
        partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
