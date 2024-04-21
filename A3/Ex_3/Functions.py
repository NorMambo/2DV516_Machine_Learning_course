
def one_vs_all(X_row, clf_list):

    scores = []

    for clf in clf_list:
        # Decision_function tells on which side of the hyperplane we are
        scores.append(clf.decision_function([X_row]))

    # Get highest value in scores found for 1 X_row by 10 classifiers
    predicted_label = max(range(len(scores)), key=lambda x: scores[x])

    return predicted_label
