import numpy as np
from sklearn.linear_model import LinearRegression

def forward_selection(X, y):
    
    # list that will contain feature columns
    feature_list = []

    # list that will contain combinations of optimal models
    models = []
    
    # number of total features per column
    n = len(y)

    # create a list of feature columns
    for i in range(0, X.shape[1]):
        column_of_features = X[:, i]
        tpl = (column_of_features, i)
        feature_list.append(tpl)

    # create a column of 1's
    Xe = np.ones((X.shape[0], 1))
    
    # count from X.shape[1], ....... ,0
    for i in reversed(range(0, X.shape[1])):

        # mse list will contain all the mse values of all the combinations of one features with the current set.
        # When the feature that contributes to the lowest mse is found, the loop starts over and the mse list is created again.
        mse_list = []
        
        for j in range(0, i + 1):
            
            # add a feature to Xe
            Xnew = np.c_[Xe, feature_list[j][0]]

            # perform linear regression on the model
            lin_reg = LinearRegression()
            lin_reg.fit(Xnew, y)
            y_pred = lin_reg.predict(Xnew)

            # calculate MSE
            MSE = np.sum(np.square(np.subtract(y, y_pred)))/n

            # create a tuple of feature +  MSE with that added feature
            tpl = (feature_list[j], MSE)
            mse_list.append(tpl)

        # find best mse value
        best_mse = min([tpl[1] for tpl in mse_list])

        # find index of feature with lowest added MSE
        best_of_round = [tpl[0][1] for tpl in mse_list if tpl[1] == best_mse]

        # select feature with best added MSE to be added to Xe for further testing
        feature_to_be_added = np.array([tpl[0][0] for tpl in mse_list if tpl[1] == best_mse])
        feature_to_be_added = feature_to_be_added.flatten()
        feature_to_be_added = feature_to_be_added.reshape(-1, 1)
        
        # add feature to Xe
        Xe = np.c_[Xe, feature_to_be_added]

        # assign column index of best feature and corresponding MSE to a tuple vairable
        best_tuple = (best_of_round, best_mse)

        # Remove the feature_column that was selected as the best from feature_list.
        # Every time this is done, the feature list shrinks by one column.
        feature_list = [tpl[0] for tpl in mse_list if tpl[1] != best_mse]

        models.append(best_tuple)

    return models