import pandas as pd
import numpy as np


class RandomSearch(object):

    def __init__(self, sampling_space):
        self.sample = sampling_space
    
    def sample_space(self, T):
        return [self.sample() for _ in range(T)]
