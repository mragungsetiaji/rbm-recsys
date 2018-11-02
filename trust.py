from __future__ import division
from os import listdir
from os.path import isfile, join
from data import getData
import numpy as np
import pdb

def trust():
    
    users = getData()
    cutOff = 0.03
    trustMat = []

    for user1 in users:
        for user2 in users:

            if user1 == user2:
                continue
        
            cnt = 0
            mo = {}

            for (m1, r1) in users[user1]:
                mo[m1] = 1
            for (m2, r2) in users[user2]:
                if mo.has_key(m2):
                    cnt +=1

            if cnt/len(users[user1]) >= cutOff and cnt/len(users[user2])>=cutOff:
                trustMat.append([user1, user2]) 
    return users, np.asarray(trustMat)

if __name__ == "__main__":
    trustMat = trust()
    pdb.set_trace()
    
