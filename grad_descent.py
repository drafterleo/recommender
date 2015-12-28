import random as rnd
import numpy as np


def random_vector(about_num, vector_length=5):
    return np.array([rnd.random() for _ in range(vector_length)])

def factorize_number(num, alpha=0.001, K=3):
    p = random_vector(int(num), vector_length=K)
    q = random_vector(int(num), vector_length=K)
    err = num - np.dot(p, q)
    while err ** 2 > 0.0001:
        #print ("[num: %f] p = %f, q = %f, p*q = %f, err = %f" % (num, p, q, p*q, err ** 2))
        print ("p =", p, "q =", q, "p*q =", np.dot(p, q), "err =", err)
        for k in range(K):
            p[k] += 2*err*q[k]*alpha
            q[k] += 2*err*p[k]*alpha
        err = num - np.dot(p, q)
    return p, q, np.dot(p, q), err ** 2


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    print(factorize_number(80))


