import os
from time import time
from numpy.random import random
import sys
from os import listdir
sys.path.append('build/' + 'lib.macosx-10.9-intel-2.7')

class LdaModel(object):

    def __init__(self, n_topic=10, alpha=0.5, beta=0.1, n_iter=10):
        self.K = n_topic
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter

    def _init_model(self, X, V):
        self.M = len(X)
        self.V = len(V)
        self.MK = [[0] * self.K] * self.M
        self.KV = [[0] * self.V] * self.K
        self.MN = [0] * self.M
        self.KN = [0] * self.K
        self.phi = [[0] * self.V] * self.K
        self.theta = [[0] * self.K] * self.M


        self.Z = [None] * self.M

        for m in xrange(self.M):
            N = len(X[m])
            self.Z[m] = [0] * N
            for n in xrange(N):
                init_topic = int(random() * self.K)
                self.Z[m][n] = init_topic
                self.MK[m][init_topic] += 1
                self.KV[init_topic][X[m][n]] += 1
                self.KN[init_topic] += 1
            self.MN[m] = N

    def fit(self, X, V):
        self._init_model(X, V)
        start = time()
        self._inference(X, V)
        print(time() - start)

    def _inference(self, X, V):
        for i in xrange(self.n_iter):
            for m in xrange(self.M):
                for n in xrange(len(X[m])):
                    new_topic = self.sample_topic(X, m, n)
                    self.Z[m][n] = new_topic
                            
    def sample_topic(self, X, m, n):
        old_topic = self.Z[m][n]
        self.MK[m][old_topic] -= 1
        v = X[m][n]
        self.KV[old_topic][v] -= 1
        self.KN[old_topic] -= 1
        self.MN[m] -= 1
        # compute a p(z_i| z_-i, w)
        p_z = [0] * self.K

        for k in xrange(self.K):
            p_z[k] = (self.KV[k][v] + self.beta) / (self.KN[k] + self.V * self.beta) * (
                self.MK[m][k] + self.alpha) / (self.MN[m] + self.K * self.alpha)
        for k in xrange(self.K - 1):
            p_z[k + 1] += p_z[k]

        u = random() * p_z[self.K - 1]
        new_topic = 0
        for k in xrange(self.K):
            new_topic = k
            if p_z[k] > u:
                break
        self.MK[m][new_topic] += 1
        self.KV[new_topic][v] += 1
        self.MN[m] += 1
        self.KN[new_topic] += 1
        return new_topic

    def estimate(self):
        for k in xrange(self.K):
            for v in xrange(self.V):
                self.phi[k][v] = (
                    self.KV[k][v] + self.beta) / (self.KN[k] + self.V * self.beta)

        for m in xrange(self.M):
            for k in xrange(self.K):
                self.theta[m][k] = (
                    self.MK[m][k] + self.alpha) / (self.MN[m] + self.K * self.alpha)
        return self.phi, self.theta
