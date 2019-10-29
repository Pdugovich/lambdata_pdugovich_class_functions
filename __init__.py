"""
Class version of Data Science helper functions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# sample code
ONES = pd.DataFrame(np.ones(50))
ZEROES = pd.DataFrame(np.zeros(50))
COMBINED = df.merge(ONES, ZEROES)

class Split:
    """
    Split data into train, validate, and test sets
    """

    def __init__(self, X, y, train_size=0.7, val_size=0.1,
         test_size=0.2, random_state=None, shuffle=True):
         self.X = X
         self.y = y
         self.train_size = train_size
         self.val_size = val_size
         self.test_size = test_size
         self.random_state = random_state
         self.shuffle = shuffle
    
    assert train_size + val_size + test_size == 1

    def train_test_validation_split(self):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, self.test_size, self.random_state, self.shuffle)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size = self.val_size
        )
        return X_train, X_val, X_test, y_train, y_val, y_test