import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("netflix_complete.txt")
K = 12
# TODO: Your code here
for seed in range(5):
    mixtures , post = common.init(X , K , seed)   

    # m, p, cost = kmeans.run(X , mixtures , post)
    # print (cost)
    # common.plot(X , mixtures , post , "Title")

    m, p, cost = naive_em.run(X , mixtures , post)      
    print (common.bic(X , m , cost))
    # common.plot(X , mixtures , post , "Title")

