import random as rnd
import numpy as np

def factorize_number_0(num, alpha=0.001):
    p = rnd.random() * rnd.randint(1, int(num))
    q = rnd.random() * rnd.randint(1, int(num))
    err = num - p*q
    while err ** 2 > 0.0001:
        print ("[num: %f] p = %f, q = %f, p*q = %f, err = %f" % (num, p, q, p*q, err ** 2))
        pt = p
        p += 2*err*q*alpha
        q += 2*err*pt*alpha
        err = num - p*q
    return p, q, p*q, err ** 2

def random_vector(length=5):
    return np.array([rnd.random() for _ in range(length)])

def factorize_number(num, alpha=0.001, K=3):
    p = random_vector(length=K)
    q = random_vector(length=K)
    e = num - np.dot(p, q)
    while e**2 > 0.0001:
        print ("p =", p, "q =", q, "p*q =", np.dot(p, q), "err =", e)
        for k in range(K):
            p[k] += 2*e*q[k]*alpha
            q[k] += 2*e*p[k]*alpha
        e = num - np.dot(p, q)
    return p, q, np.dot(p, q), e**2


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    print(factorize_number(80))


