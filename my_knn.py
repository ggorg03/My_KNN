import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def _euclidean_distance(slef, xs: list[int], ys : list[int]) -> float:
        if len(xs) == 0 or len(ys) == 0 or len(xs) != len(ys):
            return -1
        
        acc = 0
        for i in range(len(xs)):
            acc += (xs[i] - ys[i])**2
        
        return np.sqrt(acc)
    
    def get_neighbord_distance(self,sample):
        # get all distances from input sample
        distances = []
        for i in range(len(self.X_train)):
            distances.append((self.y_train[i],self._euclidean_distance(self.X_train[i],sample)))
            distances.sort(key=lambda x : x[1])
        
        neighbors=[]
        for i in range(self.k): # get first k samples
            # getting k next neighbors
            neighbors.append(distances[i][0])
        return neighbors

    def predict(self, samples):
        predictions=[]
        for sample in samples:
            neighbors = self.get_neighbord_distance(sample)
            # counting neighbors frequence
            neighbors_counter = {}
            for n in neighbors:
                if n in neighbors_counter.keys():
                    neighbors_counter[n] += 1
                else:
                    neighbors_counter[n] = 0
            # getting most frequence neighbor
            prediction=max(zip(neighbors_counter.values(),
                               neighbors_counter.keys()))[1]
            
            predictions.append(prediction)
        return predictions

