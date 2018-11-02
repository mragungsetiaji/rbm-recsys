from os import listdir
from os.path import isfile, join
import pdb

path = "../data/training_set/"
truncate = True
def getData():
    files = [ f for f in listdir(path) if isfile(join(path, f)) ]
    
    users = {}

    ct = 0
    userMap = {}
    userCt = 0
    for name in files:
        with open(path + '' + name, 'r') as f:
            index = int(f.readline().split(':')[0])
    
            for line in f.readlines():
                [u, r] = line.split(',')[:2]
                u,r = int(u), int(r)
                if u not in userMap:
                    userMap[u] = userCt
                    userCt += 1
                u = userMap[u]
                if u in users:
                    users[u].append((index, r))
                else:
                    users[u] = []
                    users[u].append((index, r))
            ct += 1
            if ct%2:
                print(ct)
   
    temp = {}
    for i, user in enumerate(users):
        if truncate:
            if i >= 1000:
                break
        temp[user] = users[user]
    
    print("data got")
    return temp


if __name__ == "__main__":
    users = getData()
    pdb.set_trace()
    
