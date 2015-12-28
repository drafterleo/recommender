import random

def factorize_number(num, alpha=0.001):
    p = random.random() * random.randint(1, int(num))
    q = random.random() * random.randint(1, int(num))
    err = num - p*q
    while err ** 2 > 0.0001:
        print ("[num: %f] p = %f, q = %f, p*q = %f, err = %f" % (num, p, q, p*q, err ** 2))
        p += 2*err*q*alpha
        q += 2*err*p*alpha
        err = num - p*q
    return p, q, p*q, err ** 2


if __name__ == '__main__':
    print(factorize_number(18.7))


