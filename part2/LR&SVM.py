from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx= None, resolution= 0.02):
    # setup marker generator and colormap
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red','blue', 'lightgreen','gray', 'cyan')
    cmap = ListedColormap(colors=colors[:len(np.unique(y))])

    #plot descision surface
    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z,alpha= 0.4, cmap= cmap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    #plot all samples
    X_test, y_test = X[test_idx,:], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x= X[y==cl, 0], y= X[y==cl, 1],alpha= 0.8,marker= markers[idx],c= cmap(idx),label = cl)

    #highlight test example
    if test_idx:
        plt.scatter(x= X_test[:, 0], y= X_test[:, 1], alpha= 1.0, marker='v',
                        c='black',linewidths= 1.0,label= 'test set',s= 20)

if __name__ == "__main__":

    # load the iris datasets
    iris = datasets.load_iris()
    X = iris.data[:,[2,3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.3, random_state= 0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std =sc.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(penalty= 'l2',C=1000,fit_intercept= True,max_iter= 40,multi_class= 'ovr')
    lr.fit(X_train_std, y_train)
    y_pred = lr.predict(X_test_std)
    from sklearn.metrics import accuracy_score
    print('Accuracy: %.3f'% accuracy_score(y_test,y_pred))
    #np.set_printoptions(precision= 3)
    np.set_printoptions(formatter= {'float': '{:0.3f}'.format})
    print('predict probability:',lr.predict_proba(X_test_std[0:3,:]))

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plt.figure(1)
    plot_decision_regions(X= X_combined_std, y = y_combined, classifier= lr,
                         test_idx= range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc= 'upper left')
    plt.title('Logistic regression classifier')
    plt.show()