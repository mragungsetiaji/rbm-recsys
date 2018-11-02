
import numpy as np
from trust import trust
from scipy.io import loadmat
import pdb

class DataHandler():
    
    def __init__(self):
        self.nUsers = 0
        self.nProd = 0
        self.nCat = 1
    
    def getStats(self):
        return self.nUsers, self.nProd, self.nCat

    def loadMatrices(self):
        # Loading Matrices from data
        users, W = trust()
        
        # Converting R and W from dictionary to array
        R = []
        for user in users:
            for (m, r) in users[user]:
                R.append([user, m, r, 0])
        R = np.asarray(R)
        pdb.set_trace()
        
        self.nUsers = max(R[:, 0])
        self.nProd = max(R[:, 1])
        
        # Selecting entries with the 6 categories given in the paper
        catId = [0]
        RSize = R.shape[0]
        
        # Choosing 70% data for training and rest for testing
        RTrain = R[:RSize*0.7]
        RTest = R[RSize*0.7:]
        
        # Making all eligible Product-Category pairs
        ones = np.ones(RTrain.shape[0])
        prodCat = dict(zip(zip(RTrain[:, 1], RTrain[:, 3]), ones))
        
        # Making the mu matrix
        mu = np.zeros(1)
        catRating = RTrain[:, 2]
        mu[0] = np.mean(catRating)
        pdb.set_trace()
        return RTrain, RTest, W, prodCat, mu
            
if __name__ == "__main__":
    data = DataHandler("../data/rating_with_timestamp.mat", "../data/trust.mat")
    RTrain, RTest, W, PFPair, mu = data.loadMatrices()
    print("done")
