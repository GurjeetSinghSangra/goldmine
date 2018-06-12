import numpy as np
import numpy.random as rng
import logging

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

from .masks import create_degrees, create_masks, create_weights, create_weights_conditional


class GaussianMADE(nn.Module):

    def __init__(self, n_inputs, n_hiddens, input_order='sequential', mode='sequential'):

        """
        Constructor.
        :param n_inputs: number of inputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param input_order: order of inputs
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        """

        super(GaussianMADE, self).__init__()

        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.mode = mode

        # create network's parameters
        self.degrees = create_degrees(n_inputs, n_hiddens, input_order, mode)
        self.Ms, self.Mmp = create_masks(self.degrees)
        self.Ws, self.bs, self.Wm, self.bm, self.Wp, self.bp = create_weights(n_inputs, n_hiddens, None)
        self.input_order = self.degrees[0]

        # Output info
        self.m = None
        self.logp = None
        self.log_likelihood = None

    def forward(self, x):

        """ Transforms x into u = f^-1(x) """

        h = x

        # feedforward propagation
        for M, W, b in zip(self.Ms, self.Ws, self.bs):
            h = F.relu(F.linear(h, torch.t(M * W), b))

        # output means
        self.m = F.linear(h, torch.t(self.Mmp * self.Wm), self.bm)

        # output log precisions
        self.logp = F.linear(h, torch.t(self.Mmp * self.Wp), self.bp)

        # random numbers driving made
        u = torch.exp(0.5 * self.logp) * (x - self.m)

        # log likelihoods
        diff = torch.sum(u ** 2 - self.logp, dim=1)
        constant = float(self.n_inputs * np.log(2. * np.pi))
        self.log_likelihood = -0.5 * (constant + diff)

        return u

    def log_p(self, x):

        """ Calculates log p(x) """

        _ = self.forward(x)

        return self.log_likelihood

    def gen(self, n_samples=1, u=None):
        """
        Generate samples from made. Requires as many evaluations as number of inputs.
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        # TODO: reformulate in pyTorch instread of numpy

        x = np.zeros([n_samples, self.n_inputs])
        u = rng.randn(n_samples, self.n_inputs) if u is None else u.data.numpy()

        for i in range(1, self.n_inputs + 1):
            self.forward(tensor(x))  # Sets Gaussian parameters: self.m and self.logp
            m = self.m.data.numpy()
            logp = self.logp.data.numpy()

            idx = np.argwhere(self.input_order == i)[0, 0]
            x[:, idx] = m[:, idx] + np.exp(np.minimum(-0.5 * logp[:, idx], 10.0)) * u[:, idx]

        return tensor(x)


class ConditionalGaussianMADE(nn.Module):
    """
    Implements a Made, where each conditional probability is modelled by a single gaussian component. The made has
    inputs theta which is always conditioned on, and whose probability it doesn't model.
    """

    def __init__(self, n_conditionals, n_inputs, n_hiddens, input_order='sequential',
                 mode='sequential'):
        """
        Constructor.
        :param n_conditionals: number of (conditional) inputs theta
        :param n_inputs: number of inputs X
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param output_order: order of outputs
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        """

        super(ConditionalGaussianMADE, self).__init__()

        # save input arguments
        self.n_conditionals = n_conditionals
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.mode = mode

        # create network's parameters
        self.degrees = create_degrees(n_inputs, n_hiddens, input_order, mode)
        self.Ms, self.Mmp = create_masks(self.degrees)
        self.Wx, self.Ws, self.bs, self.Wm, self.bm, self.Wp, self.bp = create_weights_conditional(n_conditionals,
                                                                                                   n_inputs,
                                                                                                   n_hiddens, None)
        self.input_order = self.degrees[0]

        # Output info
        self.m = None
        self.logp = None
        self.log_likelihood = None

    def forward(self, theta, x):

        """ Transforms theta, x into u = f^-1(x | theta) """

        # First hidden layer
        h = F.relu(F.linear(theta, torch.t(self.Wx)) + F.linear(x, torch.t(self.Ms[0] * self.Ws[0]), self.bs[0]))

        # feedforward propagation
        for M, W, b in zip(self.Ms[1:], self.Ws[1:], self.bs[1:]):
            h = F.relu(F.linear(h, torch.t(M * W), b))

        # output means
        self.m = F.linear(h, torch.t(self.Mmp * self.Wm), self.bm)

        # output log precisions
        self.logp = F.linear(h, torch.t(self.Mmp * self.Wp), self.bp)

        # random numbers driving made
        u = torch.exp(0.5 * self.logp) * (x - self.m)

        # log likelihoods
        diff = torch.sum(u ** 2 - self.logp, dim=1)
        constant = float(self.n_inputs * np.log(2. * np.pi))
        self.log_likelihood = -0.5 * (constant + diff)

        return u

    def log_p(self, theta, x):

        """ Calculates log p(x) """

        _ = self.forward(theta, x)

        return self.log_likelihood

    def gen(self, theta, n_samples=1, u=None):
        """
        Generate samples from made. Requires as many evaluations as number of inputs.
        :param theta: conditionals
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        # TODO: reformulate in pyTorch instread of numpy

        x = np.zeros([n_samples, self.n_inputs])
        u = rng.randn(n_samples, self.n_inputs) if u is None else u.data.numpy()

        for i in range(1, self.n_inputs + 1):
            self.forward(tensor(theta), tensor(x))
            m = self.m.data.numpy()
            logp = self.logp.data.numpy()

            idx = np.argwhere(self.input_order == i)[0, 0]
            x[:, idx] = m[:, idx] + np.exp(np.minimum(-0.5 * logp[:, idx], 10.0)) * u[:, idx]

        return tensor(x)
