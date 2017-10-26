import numpy as np
import numpy.random as rnd
def digitize(data, bins):
    
    #steps = []
    res = np.zeros(data.size, dtype=int)
    for n,x in enumerate(data.flatten()):
        i = 0
        j = len(bins)-1
        #print "value", x
        c = 0
        while j - i > 1:
            pivot = max(1, int((j - i)/2 )) + i
            #print pivot, bins[pivot]
            #print "indices",i,j, pivot
            #print "values",'?', bins[i], bins[j], bins[pivot]
            
            if x >= bins[pivot]:
                i = pivot
            else:
                j = pivot
            c += 1
        if x >= bins[j]:
            res[n] = j
        else:
            res[n] = i

        #print x,"->", res[n], "in {0} steps".format(c)
        #steps.append(c)
        
    #print np.array(steps).mean(), "average steps"
    return res.reshape(data.shape)
        
        
bins = [-1,1,3,5,10,15,20]

data = np.array([0,2,4,6,7,8,9,10,16,30,100, -1])
#data = rnd.random(100) * 22 - 1

res = digitize(data, bins)
res2 = np.digitize(data, bins) - 1 

print res
print res2
print res == res2
