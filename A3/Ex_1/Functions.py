import numpy as np
import matplotlib.pyplot as plt

def plot_boundary(clf, X, y):
    
    plt.scatter(X[:, 0], X[:, 1], c=y, s=2)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                         np.linspace(ylim[0], ylim[1], 100))
    
    pred = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax.contourf(xx, yy, pred, alpha=0.5, cmap='tab10')
    plt.show()