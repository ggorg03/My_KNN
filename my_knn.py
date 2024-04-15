import numpy as np
from tqdm import tqdm

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def _euclidean_distance(slef, xs: list[int], ys : list[int]) -> float:
        acc = 0
        for i in range(len(xs)):
            acc += (xs[i] - ys[i])**2
        
        return np.sqrt(acc)
    
    def _get_k_nearst_neighbord(self,sample):
        # get all distances from input sample
        distances = []
        for i in range(len(self.x_train)):
            distances.append((self.y_train[i],self._euclidean_distance(self.x_train[i],sample)))
            distances.sort(key=lambda x : x[1])
        # getting k nearest neighbors
        neighbors=[]
        for i in range(self.k): # get first k samples
            neighbors.append(distances[i][0])
        return neighbors

    def predict(self, samples):
        predictions=[]
        for sample in tqdm(samples, desc="predict progress"):
            neighbors = self._get_k_nearst_neighbord(sample)
            # counting nearest neighbors frequence
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

# Testing model with mnist dataset
if __name__=='__main__':
    # getting data
    mnist_dataset = load_digits()
    x = mnist_dataset.data
    y = mnist_dataset.target
    # split train test datasets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # testing my KNN
    knn = KNN()
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    # getting some metrics
    conf_matrix = confusion_matrix(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    ## PRINTING RESULTS
    print("confusion matrix", conf_matrix, sep='\n')
    print(f'{accuracy=}')