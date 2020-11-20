import pickle
import numpy as np


class Evaluator():
    def __init__(self, file_data='../data/kemmeren/orig.p'):
        self.data_int = None
        self.file_data = file_data

    def set_data_int(self, data_int):
        self.data_int = data_int

    def penalty_absolute(self, order, int_ids=None, return_ratio=False):
        """
        Returns the average intervention value per violated relation by some variable order
        Params:
            order: variable order as list of ids
            int_ids: if given, a subset of intervention ids used for the order
            file_data: the intervention data is loaded from here
            return_ratio: set to True to also return the ratio of penalty : average value
        """
        # Load data
        if self.data_int is None:
            data = pickle.load(open(self.file_data, 'rb'))
            self.data_int = data[1][data[2]] # interv x interv data

        N = len(order)
        # if int_ids is None: int_ids = list(range(N))
        # D = self.data_int[int_ids][:,int_ids]
        
        # Compute penalty
        D = self.data_int[order][:,order]
        score = np.sum(abs(np.triu(D, k=1))) / (.5*(N**2-N))
        if return_ratio:
            avg = np.sum(abs(np.triu(D, k=1)) + abs(np.tril(D,k=-1))) / (N**2-N)
            return score, score / avg
        else:
            return score
