
from __future__ import division
import numpy as np
import pdb
import json
from data import getData
import copy

N_IT = 1
ETA = 0.001

users = {}
userMovies = {}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class RBM():

    def __init__(self, data):
        
        self.F = 100
        self.K = 5
        self.m = 0

        #self.m = data.shape[0]   # No of movies rated by user
        for i, u in enumerate(users):
            temp = [movieId for (movieId, rating) in data[u]]
            self.m = max(self.m, max(i for i in temp) + 1)
        
        self.h = np.random.rand(self.F) - 0.5
        self.featureBias = np.random.rand(self.F) - 0.5
        self.movieBias = np.random.rand(self.m, self.K) - 0.5
        self.w = np.random.rand(self.F, self.m, self.K) - 0.5
        self.data = data
        
    def train(self):
        for it in range(N_IT):
            for u in users:
                data = copy.deepcopy(self.data[u])

                w = self.getW(userMovies[u])
                posAssociations, self.h = self.fwdProp(data, userMovies[u])
                visibleProb = self.bwdProp(self.h, userMovies[u])
                negAssociations, temp = self.fwdProp(visibleProb, userMovies[u])

                w += ETA * (posAssociations - negAssociations) / len(userMovies[u]) 
                self.setW(userMovies[u], w)
                error = np.sum((data - visibleProb) ** 2)
                error = np.sqrt(error/len(data))
                print(it, u, error)


    def getW(self, movies):

        a = np.zeros((self.F, 1, self.K))
        for m in movies:
            a = np.concatenate((a, np.expand_dims(self.w[:,m,:], axis=1)), axis=1)
        return a[:,1:,]

    def setW(self, movies, w):
        
        it = 0
        for m in movies:
            self.w[:, m, :] = w[:, it, :]
            it += 1
        
    def getMovieBias(self, movies):
        
        a = np.zeros((1, self.K))
        for m in movies:
            a = np.concatenate((a, np.expand_dims(self.movieBias[m,:], axis=0)), axis=0)
        return a[1:,]

    def fwdProp(self, inp, movies):
        hiddenUnit = np.copy(self.featureBias)
        for j in range(self.F):
            hiddenUnit[j] += np.tensordot(inp, self.getW(movies)[j])
        hiddenProb = sigmoid(hiddenUnit)
        hiddenStates = hiddenProb > np.random.rand(self.F)
        hiddenAssociations = np.zeros((self.F, len(movies), self.K)) 
        for j in range(self.F):
            hiddenAssociations[j] = hiddenProb[j] * inp
        return hiddenAssociations, hiddenStates

    def bwdProp(self, inp, movies):
        visibleUnit = self.getMovieBias(movies)
        for j in range(self.F):
            visibleUnit += inp[j] * self.getW(movies)[j]
        visibleProb = sigmoid(visibleUnit)
        return visibleProb

    def predictor(self, movieId, userId):

        w = self.getW(userMovies[userId])
        
        #making predictions part Vq not given
        data = copy.deepcopy(self.data[userId])
        probs = np.ones(5)
        
        mx, index = -1, 0

        for i in range(5):
            calc = 1.0
            for j in range(self.F):
                temp = np.tensordot(data, self.getW(userMovies[userId])[j]) + self.featureBias[j]
                temp = 1.0 + np.exp(temp)
                calc *= temp
            probs[i] = calc

            if mx < probs[i]:
                index = i
                mx = probs[i]

        return index

def demo():
    data = [[0,0,1,0,0], [0,1,0,0,0], [0,0,0,0,1]]
    data = np.asarray(data)
    rbm = RBM(data)
    rbm.train()

## Test Model
if __name__ == '__main__': 
    
    rawData1 = getData()
    rawData = {}
    ct = 0
    for i in rawData1.keys():
        rawData[i] = rawData1[i]
        ct += 1
        if ct >= 1000:
            pdb.set_trace()
            break
    print('Data done')
    users = rawData.keys()
    users.sort()
    data = {}
    for i, u in enumerate(users):
        userMovies[u] = [movieId for (movieId, rating) in rawData[u]]
        data[u] = [[0]*(rat-1) + [1] + [0]*(5-rat) for (movId, rat) in rawData[u]]
        data[u] = np.asarray(data[u])

    rbm = RBM(data)
    print('Training DONE!')
    rbm.train()
    pdb.set_trace()
