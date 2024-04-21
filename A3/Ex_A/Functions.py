import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def first_plot(clf, X_s, support_vectors):
    x_min, x_max = X_s[:, 0].min() - 1, X_s[:, 0].max() + 1
    y_min, y_max = X_s[:, 1].min() - 1, X_s[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))

    pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    pred = pred.reshape(xx.shape)

    plt.contourf(xx, yy, pred, alpha=0.4, cmap=cm.jet)

    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=6,
                linewidth=1, edgecolors='k')

    plt.show()

def second_plot(clf, X, y):
    
    plt.scatter(X[:, 0], X[:, 1], c=y, s=2)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                         np.linspace(ylim[0], ylim[1], 100))
    
    pred = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax.contourf(xx, yy, pred, alpha=0.5, cmap='winter')
    plt.show()