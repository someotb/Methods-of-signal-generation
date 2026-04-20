import numpy as np
from matplotlib import pyplot as plt


def bpsk_mapper(bits):
    symbols = complex(2.0 - 1.0 * bits, 2.0 - 1.0 * bits) / np.sqrt(2.0)
    return symbols


# Task 1
bits = np.random.randint(0, 2, 100)
symbols = bpsk_mapper(bits)
