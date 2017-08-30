import numpy as np

from sklearn import datasets
import matplotlib.pyplot as plt


def GetDatasets():
    iris = datasets.load_iris()
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

class AdalineGD(object):
    """ Adaptive linear neuron classifier.
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
    errors_: list
        Number of misclassifications in every epoch.
    """

    def __init__(self, eta=0.01, n_iter=50):
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
        self.cost_= []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


if __name__ == "__main__":
    X, y = GetDatasets()
    """Select the before 100 numbers and
        the first sepal length and third petal length features
    """
    X = X[0:100, [0, 2]]
    y = y[0:100]
    y = np.where(y == 0, 1, -1)
    """
    plt.figure(1)
    f1 = plt.scatter(X[:50, 0], X[:50, 1], c='red', edgecolors='red', marker='o')
    f2 = plt.scatter(X[50:100, 0], X[50:100, 1], c='blue', edgecolors='blue', marker='*')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('distribution of source features')
    plt.legend((f1, f2), ('setosa', 'versicolor'), loc='upper left', ncol=1, fontsize=8)
    plt.show()
    
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.figure(2)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('number of misclassifications')
    plt.title('errors rate curve')
    plt.show()

    plt.figure(3)
    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.title('decision boundary')
    plt.show()
    """
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize= (8,4))
    ada1 = AdalineGD(n_iter= 10,eta= 0.01).fit(X,y)
    ax[0].plot(range(1, len(ada1.cost_)+1),
               np.log(ada1.cost_),marker= 'o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')
    ada2 = AdalineGD(n_iter= 10, eta= 0.0001).fit(X,y)
    ax[1].plot(range(1, len(ada2.cost_)+1),
               ada2.cost_,marker= 'o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()
    #print (ada1.cost_,"\n", ada2.cost_)
