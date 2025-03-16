"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def add(x, y):
    return x + y

def mul(x, y):
    return x * y

def id(x):
    return x

def neg(x):
    return -x

def lt(x, y):
    return 1.0 if x < y else 0.0

def eq(x, y):
    return 1.0 if x == y else 0.0

def max(x, y):
    return x if x >= y else y

def is_close(x, y):
    return 1.0 if math.fabs(x - y) < math.exp(-6) else 0.0

# https://en.wikipedia.org/wiki/Sigmoid_function
def sigmoid(x):
    return (
        1.0 / (1 + math.exp(-x))
    )

# https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
def relu(x):
    return x if x > 0 else 0

def log(x):
    return math.log(x)

def exp(x):
    return math.pow(math.e, x)

def inv(x):
    return 1.0 / x

def log_back(a, b):
    return b / a

def inv_back(a, b):
    return -b / (a ** 2)

def relu_back(a, b):
    return b if a > 0 else 0

## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

# each x -> f(X)
def map(fn):
    def process(ls):
        arr = []
        for l in ls:
            arr.append(fn(l))
        return arr

    return process


# each x, y -> fn(x, y)
def zipwith(fn):
    def process(ls1, ls2):
        arr = []
        for i in range(len(ls1)):
            arr.append(fn(ls1[i], ls2[i]))
        return arr
    return process


def reduce(fn, start):
    def process(ls):
        arr = start
        for l in ls:
            arr = fn(l, arr)
        return arr

    return process

def negList(ls):
    return map(neg)(ls)

def addLists(ls1, ls2):
    return zipwith(add)(ls1, ls2)

def sum(ls):
    return reduce(add, 0.0)(ls)

def prod(ls):
    return reduce(mul, 1.0)(ls)