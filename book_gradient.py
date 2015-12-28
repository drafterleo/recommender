import random
import math

def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def vector_subtract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def sum_of_squares(v):
    return dot(v, v)

def magnitude(v):
    return math.sqrt(sum_of_squares(v))

def distance(v, w):
    return magnitude(vector_subtract(v, w))

def step(v, direction, step_size):
    print(v, direction, step_size)
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]

def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]

# pick a random starting point
v = [random.randint(-10,0) for i in range(1)]
tolerance = 0.0000001

while True:
    gradient = sum_of_squares_gradient(v) # compute the gradient at v
    next_v = step(v, gradient, -0.01)     # take a negative gradient step
    if distance(next_v, v) < tolerance:   # stop if we're converging
        break
    v = next_v                            # continue if we're not
