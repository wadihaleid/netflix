from operator import pos
import numpy as np
import em
import common
import naive_em


X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

# TODO: Your code here
# mixtures , post = common.init(X , K , seed)
# new_mixture , new_post , new_ll = naive_em.run(X , mixtures , post)

# print(new_mixture)
# print(new_ll)

mixtures , post = common.init(X , K , seed)
X_complete = em.fill_matrix(X , mixtures)
print(X_gold)
print(X_complete)



