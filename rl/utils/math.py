import numpy as np


def distance(a, b):
    return np.linalg.norm(a -b)


def uniform_sample(buffer, num_samples):
    return buffer[np.random.randint(len(buffer), size=min(num_samples, len(buffer)))]