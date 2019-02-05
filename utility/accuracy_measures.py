import numpy as np

def compute_precision(confusion_matrix):
    sum = confusion_matrix['tp'] + confusion_matrix['fp']
    if sum == 0:
        return np.nan
    return confusion_matrix['tp'] / sum

def compute_recall(confusion_matrix):
    sum = confusion_matrix['tp'] + confusion_matrix['fn']
    if sum == 0:
        return np.nan
    return confusion_matrix['tp'] / sum

def compute_specificity(confusion_matrix):
    sum = confusion_matrix['tn'] + confusion_matrix['fp']
    if sum == 0:
        return np.nan
    return confusion_matrix['tn'] / sum

def compute_false_positive_rate(confusion_matrix):
    sum = confusion_matrix['tn'] + confusion_matrix['fp']
    if sum == 0:
        return np.nan
    return confusion_matrix['fp'] / sum

def compute_negative_prediction_value(confusion_matrix):
    sum = confusion_matrix['tn'] + confusion_matrix['tp']
    if sum == 0:
        return np.nan
    return confusion_matrix['tn'] / sum 

def compute_accuracy(confusion_matrix):
    return (confusion_matrix['tp'] + confusion_matrix['tn']) / confusion_matrix['total']

def compute_f1(confusion_matrix):
    p = compute_precision(confusion_matrix)
    r = compute_recall(confusion_matrix)
    return 2 * (( p * r ) / (p + r))

def compute_mcc(confusion_matrix):
    top = confusion_matrix['tp'] * confusion_matrix['tn'] - confusion_matrix['fp'] * confusion_matrix['fn']
    bottom = confusion_matrix['tp'] + confusion_matrix['fn']
    bottom *= confusion_matrix['tp'] + confusion_matrix['fp']
    bottom *= confusion_matrix['tn'] + confusion_matrix['fp']
    bottom *= confusion_matrix['tn'] + confusion_matrix['fn']
    bottom = np.sqrt(bottom)
    if bottom == 0:
        return np.nan
    return top / bottom