"""Collection of the core mathematical operators used throughout the code base."""

import math
import operator
import builtins

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


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> bool:
    return x < y


def eq(x: float, y: float) -> bool:
    return x == y


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def relu(x: float) -> float:
    return max(0., x)


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    return d / x


def inv(x: float) -> float:
    return 1 / x


def inv_back(x: float, d: float) -> float:
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.


# ## Task 0.3

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


def map(func: Callable, iterable: Iterable) -> Iterable:
    return builtins.map(func, iterable)


def zipWith(func: Callable, a: Iterable, b: Iterable) -> Iterable:
    return map(lambda pair: func(pair[0], pair[1]), zip(a, b))


def reduce(func: Callable[[float, float], float], iterable: Iterable[float]) -> float:
    iterator = iter(iterable)
    result = next(iterator)
    for item in iterator:
        result = func(result, item)
    return result


def negList(l: Iterable[float]) -> Iterable[float]:
    return map(lambda x: -x, l)


def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    return zipWith(operator.add, a, b)


def sum(l: Iterable[float]) -> float:
    try:
        return reduce(operator.add, l)
    except StopIteration:
        return 0.0


def prod(l: Iterable[float]) -> float:
    try:
        return reduce(operator.mul, l)
    except StopIteration:
        return 1.0
