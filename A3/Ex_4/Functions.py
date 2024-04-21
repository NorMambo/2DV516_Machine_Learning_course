import numpy as np
import matplotlib.pyplot as plt

def plot_boundary(clf_list, X_s):
    x_min, x_max = X_s[:, 0].min() - 1, X_s[:, 0].max() + 1
    y_min, y_max = X_s[:, 1].min() - 1, X_s[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    plt.figure(figsize=(15, 12))

    prediction_list = []
    for clf, n in zip(clf_list, range(len(clf_list))):
        ax = plt.subplot(10, 10, n + 1)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        prediction_list.append(pred)

        pred = pred.reshape(xx.shape)

        ax.contourf(xx, yy, pred, alpha=0.4, colors="g")
    plt.show()
    
    majority_vote_pred = []
    # Majority vote
    for i in range(18000):
        one = 0
        zero = 0
        for j in range(len(prediction_list)):
            if prediction_list[j][i] == 1:
                one += 1
            else:
                zero += 1
        if one > zero:
            majority_vote_pred.append(1)
        else:
            majority_vote_pred.append(0)
        
    majority_vote_pred = np.array(majority_vote_pred)
    majority_vote_pred = majority_vote_pred.reshape(xx.shape)
    plt.figure(figsize=(15, 12))
    plt.contourf(xx, yy, majority_vote_pred, alpha=0.4, colors='r')
    plt.show()

    