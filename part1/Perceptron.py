import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def GetDatasets():
    iris= datasets.load_iris()
    X, y = iris.data, iris.target
    return X, y


class Perceptron(object):
    """ Perceptron classifier.
    Parameters
    ------
    eta: float
        Learning rate （between 0.0 and 1.0)
    n_iter: int
        Passes over the training datasets.
    Attributes
    -----------
    w_: 1d-array
        Weights after fitting.
    errors: list
        Number of misclassifications in every epoch.
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.
        Parameters
        -----------------
        X: {array-like},shape = [n_samples,n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of featrues
        y: array-like, shape = [n_sample]
            Target values
        Returns
        -----------------------
        self: object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
if __name__ == "__main__":
    X,y = GetDatasets()
    """Select the before 100 numbers and
        the first sepal length and third petal length features
    """
    X = X[0:100,[0, 2]]
    y = y[0:100]
    y = np.where(y == 0, 1, -1)
    f1 = plt.scatter(X[:50,0], X[:50,1],c= 'red', edgecolors='red', marker= 'o')
    f2 = plt.scatter(X[50:100,0], X[50:100,1], c= 'blue', edgecolors='blue', marker= '*')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')

    plt.legend((f1,f2),('setosa','versicolor'),loc ='upper left',ncol = 1, fontsize = 8)
    plt.show()

    ppn = Perceptron(eta = 0.1, n_iter = 10)
    ppn.fit(X, y)
    plt.figure(2)
    plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('number of misclassifications')
    plt.show()



