from math import *

def loadRatingMatrixSparse(filename):
    mat = {}
    userN = 0
    itemN = 0
    sumRate = 0
    cnt = 0
    for line in open(filename):
        tokens = line.split()
        user = int(tokens[0]) - 1 # from 1-based to 0-based
        if user < 0:
            print '[ERROR] user index < 0'
        item = int(tokens[1]) - 1 # from 1-based to 0-based
        if item < 0:
            print '[ERROR] item index < 0'
        rate = float(tokens[2])
        # let's ignore the time
        
        userN = max(userN, user + 1)
        itemN = max(itemN, item + 1)
        
        if user not in mat:
            mat[user] = {}
        mat[user][item] = rate
        
        sumRate += rate
        cnt += 1
    return (mat, userN, itemN, sumRate / cnt)
    
(train, trainUserN, trainItemN, trainAvg) = loadRatingMatrixSparse('ml100k/ua.base')
(test, testUserN, testItemN, testAvg) = loadRatingMatrixSparse('ml100k/ua.test')

userN = max(trainUserN, testUserN)
itemN = max(trainItemN, testItemN)

print 'users =', userN
print 'items =', itemN
print 'avg =', trainAvg

print 'userFeat =', userN + itemN
print 'itemFeat =', itemN

out = open('userFeatMat', 'w')
for user in xrange(userN):
    feat = []
    feat.append((user, 1))
    if user in train:
        coef = 1.0 / sqrt(len(train[user]))
        for (item, rate) in train[user].items():
            feat.append((item + userN, coef))
    out.write(str(user) + ' ' + str(len(feat)))
    for (user, coef) in feat:
        out.write(' ' + str(user) + ':' + str(coef))
    out.write('\n')
out.close()

out = open('itemFeatMat', 'w')
for item in xrange(itemN):
    out.write(str(item) + ' 1 ' + str(item) + ':1\n')
out.close()

out = open('train', 'w')
for (user, dict) in train.items():
    for (item, rate) in dict.items():
        out.write(str(user) + '\t' + str(item) + '\t' + str(rate) + '\n')
out.close()

out = open('test', 'w')
for (user, dict) in test.items():
    for (item, rate) in dict.items():
        out.write(str(user) + '\t' + str(item) + '\t' + str(rate) + '\n')
out.close()

