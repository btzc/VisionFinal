from sklearn.metrics import precision_score, recall_score, accuracy_score

#both the accuracy_score, precision_score, and recall_score methods are from https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

def evaluate(output):
    metrics = {}
    y_true = [ex[0][:-5] == ex[1][:-5] for ex in output]
    y_pred = [ex[2] for ex in output]
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    return metrics
